"""Controller-driven L3 -> L1 prefetch through the unified-cache external linker.

A prefetch takes the path a real request takes up to its forward, without
the forward: ``match_kv_cache`` (tree walk plus the linker's store lookup,
reduced across ranks), ``linker.load_back`` (the hit lands in slots private
to the request), ``start_layer_wise_loading``. The load thread then fills
the slots layer by layer while the scheduler loop keeps running; once every
rank's load has landed, the request is released with ``checkpoint=True`` and
the tail joins the radix tree exactly as a finished request's KV would.

The handler runs on every TP rank with the same arguments (control requests
are broadcast), and the only cross-rank decisions it takes -- the match and
the landed verdict -- are reduced, so the ranks never diverge on whether the
tail is inserted.

Ingress. A :class:`PrefetchReqInput` arrives on the scheduler's tokenizer
input zmq socket (``TokenizerManager.prefetch`` sends it) or on the
``--hint-zmq-endpoint`` socket; the group leader pulls it in
``SchedulerRequestReceiver.recv_requests`` and broadcasts it with the batch
to the other ranks of its KV cache, and the type dispatcher routes it here.
Only the leader owns the reply sockets, so only its reply leaves the process.

Failure discipline. Everything that can be decided from the hint alone
(validation, the refusals) is decided before any collective and answered
with ``error``; it is identical on every rank. Past that point the ranks
take collectives together, and a failure propagates exactly as a request's
would: swallowing it on one rank would leave the others waiting in the next
all-reduce. Rank-local conditions that may legitimately differ (a request
row running out) are reduced before they are acted on.

Per page, the path a hit takes:

* ``match_kv_cache`` walks the tree for the device prefix, then
  ``UnifiedCacheLinkerWrapper.match`` hashes the device-uncached tail per
  page (``get_storage_hash_str``, chained from the last device page), builds a
  LOOKUP ``PoolTransfer`` per component and calls ``linker.lookup`` ->
  ``MooncakeStore.batch_exists_v2`` for the restorable prefix lengths; the
  lengths are intersected across ranks (all-reduce MIN) and the hit is kept
  as an ``ExternalCacheHitMarker``.
* ``load_back`` builds a LOAD ``PoolTransfer`` per component: the FULL
  component evicts for any shortfall and takes device slots for the hit pages;
  ``linker.load`` resolves the transfers and queues them under the rid.
* ``start_bulk_loading`` moves the queued transfers to the load thread with
  a transfer-ready event. No forward waits on a prefetch, so the thread opens
  a Mooncake get session per pool and issues ranged gets that carry every
  layer of every page (``bulk_load_layers_per_get`` chunks them) straight
  into the registered device pages, completing the layer futures
  (``LayerWiseLoadCounter``) as each get lands.
* When every rank agrees the pages landed, the request is released with
  ``checkpoint=True``: ``UnifiedRadixCache.checkpoint_into_tree`` inserts
  ``[0, prefix_len)`` from the request row (the device prefix dedups against
  the existing path), and the write-through chain skips nodes the store
  already holds. From then on a matching request is a device hit.
"""

from __future__ import annotations

import logging
import time
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import torch

from sglang.srt.managers.io_struct import PrefetchReqInput, PrefetchReqOutput
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )

logger = logging.getLogger(__name__)

# Error prefix of a hint answered by an engine that serves another KV cache;
# the tokenizer-side sender filters these out of the fan-out replies.
NOT_SERVED_ERROR = "Hint targets dp_rank"


@dataclass
class _PendingPrefetch:
    recv_req: PrefetchReqInput
    req: Req
    counter_index: int
    device_hit_tokens: int
    loaded_tokens: int
    started_at: float
    # Set once every rank agreed whether the store read landed.
    landed: Optional[bool] = None


class SchedulerPrefetchWrapper:
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        send_output: Callable[[PrefetchReqOutput, PrefetchReqInput], None],
        served_dp_rank: int = 0,
    ) -> None:
        self._scheduler = scheduler
        # Answers a prefetch through the scheduler's control-reply path.
        self._send_output = send_output
        # The KV cache this scheduler serves; see PrefetchReqInput.dp_rank.
        self._served_dp_rank = served_dp_rank
        self._pending: Optional[_PendingPrefetch] = None

    def has_pending(self) -> bool:
        return self._pending is not None

    @property
    def _linker(self) -> Optional[UnifiedCacheLinkerWrapper]:
        return getattr(self._scheduler.tree_cache, "linker", None)

    # ---- dispatch: refuse, else match, allocate, start the load ----

    def handle(self, recv_req: PrefetchReqInput) -> Optional[PrefetchReqOutput]:
        started_at = time.perf_counter()
        try:
            problem = self._refusal(recv_req)
        except Exception as error:
            # Deciding on the hint alone must not take the loop down.
            logger.exception("Prefetch validation failed (rid=%s)", recv_req.rid)
            problem = f"Prefetch rejected: {error!r}"
        if problem is not None:
            return self._failure(recv_req, started_at, problem)
        return self._handle(recv_req, started_at)

    def _refusal(self, recv_req: PrefetchReqInput) -> Optional[str]:
        """Why the hint is not served, decided from the hint and uniform
        configuration only: identical on every rank, no collectives."""
        problem = self._validate(recv_req)
        if problem is not None:
            return problem
        if self._linker is None:
            return "The unified-cache external linker is not enabled."
        if self._pending is not None:
            return "Another prefetch is already in progress."
        components = getattr(self._scheduler.tree_cache, "components", {})
        if ComponentType.MAMBA in components:
            # A finished insert keeps a Mamba state only from a tracked
            # checkpoint, which a request that never ran a forward lacks.
            return "Prefetch is not supported with Mamba trees."
        if ComponentType.SWA in components:
            # The sliding-window pages of a prefix nobody decoded through are
            # tombstoned on insert; the prefetched prefix would be unusable.
            return "Prefetch is not supported with SWA trees."
        if getattr(self._linker.cache_linker, "mla_broadcaster", None) is not None:
            # The dedup broadcast is driven by the forward's per-layer waits.
            return "Prefetch is not supported with --enable-linker-mla-dedup."
        return None

    def _validate(self, recv_req: PrefetchReqInput) -> Optional[str]:
        if not isinstance(recv_req.rid, str) or not recv_req.rid:
            return "rid must be a non-empty string."
        dp_rank = recv_req.dp_rank
        if dp_rank is not None and (
            isinstance(dp_rank, bool) or not isinstance(dp_rank, int)
        ):
            return "dp_rank must be an int or null."
        if (dp_rank or 0) != self._served_dp_rank:
            return (
                f"{NOT_SERVED_ERROR} {dp_rank or 0}; this engine serves "
                f"dp_rank {self._served_dp_rank}."
            )
        token_ids = recv_req.token_ids
        if not isinstance(token_ids, (list, tuple)) or not token_ids:
            return "token_ids must be a non-empty list of ints."
        max_tokens = self._max_tokens()
        if len(token_ids) > max_tokens:
            return f"token_ids longer than the prefetch limit ({max_tokens})."
        try:
            ids = np.asarray(token_ids)
        except (TypeError, ValueError):
            return "token_ids must be a non-empty list of ints."
        if ids.ndim != 1 or ids.dtype.kind not in "iu":
            return "token_ids must be a non-empty list of ints."
        vocab_size = getattr(
            getattr(self._scheduler, "model_config", None), "vocab_size", None
        )
        low, high = int(ids.min()), int(ids.max())
        if low < 0 or (vocab_size is not None and high >= vocab_size):
            return f"token id {low if low < 0 else high} is outside the vocabulary."
        return None

    def _max_tokens(self) -> int:
        """--prefetch-max-tokens, else the context length."""
        from sglang.srt.runtime_context import get_memory

        cap = get_memory().prefetch_max_tokens
        if cap is None:
            cap = getattr(
                getattr(self._scheduler, "model_config", None), "context_len", None
            )
        return cap if cap is not None else 1 << 31

    def _budget_tokens(self) -> int:
        """Page-aligned tokens a prefetch may take from the KV pool without
        eating the running batch's decode headroom: free plus evictable minus
        the reserve prefill admission keeps for the running requests (the
        same estimate PrefillAdder uses), reduced MIN so every rank truncates
        the hit identically."""
        from sglang.srt.managers.schedule_policy import running_request_token_reserve

        scheduler = self._scheduler
        budget = scheduler.token_to_kv_pool_allocator.create_prefill_budget(
            scheduler.tree_cache
        )
        running = scheduler.running_batch
        if running is not None and not running.is_empty():
            ratio = scheduler.new_token_ratio_tracker.current
            budget.total_offset += sum(
                running_request_token_reserve(req, ratio) for req in running.reqs
            )
        page = self._scheduler.tree_cache.page_size
        tokens = max(0, int(budget.remaining_total)) // page * page
        reduced = torch.tensor([tokens], dtype=torch.int64)
        scheduler.tree_cache._all_reduce_attn_groups(
            reduced, torch.distributed.ReduceOp.MIN
        )
        return int(reduced.item())

    def _handle(
        self, recv_req: PrefetchReqInput, started_at: float
    ) -> Optional[PrefetchReqOutput]:
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.mem_cache.common import match_kv_cache
        from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
        from sglang.srt.sampling.sampling_params import SamplingParams

        linker = self._linker
        scheduler = self._scheduler
        tree_cache = scheduler.tree_cache

        # Headroom first: the hit is matched and loaded only up to what the
        # pool can take without starving decode. Conservative: the device hit
        # inside the prefix counts against it too.
        budget = self._budget_tokens()
        if budget == 0:
            return self._failure(recv_req, started_at, "No KV headroom for a prefetch.")
        token_ids = recv_req.token_ids[:budget]

        req = Req(
            f"prefetch-{recv_req.rid}",
            "",
            array("q", token_ids),
            SamplingParams(max_new_tokens=0),
        )
        req.refresh_fill_ids()
        # No `input_len - 1` cap: the prefix is prefetched whole, last page
        # included.
        match_result = match_kv_cache(
            req,
            tree_cache,
            req.full_untruncated_fill_ids,
            cow_mamba=False,
            max_prefix_len=None,
        )
        if match_result.cache_protected_len is None:
            req.kv.protect_prefix(req.prefix_len)
        device_hit_tokens = req.prefix_len
        linker.mark_timing_kind(req.rid, "prefetch")

        if not linker.has_hit(req.rid):
            linker.release_request(req.rid)
            return PrefetchReqOutput(
                rid=recv_req.rid,
                matched_tokens=device_hit_tokens,
                loaded_tokens=0,
                elapsed_ms=self._elapsed_ms(started_at),
            )

        storage = kv_storage_of(
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
        )
        try:
            rows = storage.acquire([req])
        except Exception as error:  # a state sub-pool out of slots
            rows = None
            logger.warning("Prefetch could not bind a request row: %s", error)
        # A row may run out on one rank only; decide together.
        if not self._all_ranks(rows is not None):
            if rows is not None:
                storage.release_row(req)
            linker.release_request(req.rid)
            return self._failure(
                recv_req, started_at, "No free request slot for the prefetch."
            )

        # Eviction inside load_back must not take the matched device prefix.
        req.lock = tree_cache.lock(req.last_node)
        loaded_tokens, req.last_node = linker.load_back(req)
        if loaded_tokens == 0:
            self._discard(req, storage)
            return self._failure(
                recv_req,
                started_at,
                "The linker refused the load.",
                matched_tokens=device_hit_tokens + req.host_hit_length,
            )

        # The prefetch is not part of the batch the wrapper is building:
        # finish_loads after the next forward must not wait on or drop it.
        linker.inflight_load_rids.remove(req.rid)
        # A bulk load: nothing waits on layers, so the backend may fetch
        # whole pages.
        counter_index = linker.cache_linker.start_bulk_loading()

        # Register the loaded tail in the request row, as prepare_for_extend
        # would; release_kv_cache reads the row when it inserts or frees.
        req.prefix_len = device_hit_tokens + loaded_tokens
        indices = tree_cache.prefix_device_indices(req)
        linker.private_load_indices.pop(req.rid, None)
        scheduler.req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, req.prefix_len)), indices
        )
        req.kv.set_row_len(req.prefix_len)

        pending = _PendingPrefetch(
            recv_req=recv_req,
            req=req,
            counter_index=counter_index,
            device_hit_tokens=device_hit_tokens,
            loaded_tokens=loaded_tokens,
            started_at=started_at,
        )
        if counter_index < 0:
            # Nothing was queued to load: free the tail, answer now.
            pending.landed = False
            self._finish(pending, landed=False)
            return None
        self._pending = pending
        return None

    def _discard(self, req: Req, storage) -> None:
        """Give back what a request that will not load holds: the match lock,
        the hit marker and timing record, private slots, the row."""
        linker = self._linker
        if req.lock is not None:
            self._scheduler.tree_cache.unlock(req.lock)
            req.lock = None
        linker.release_request(req.rid)
        linker.private_load_indices.pop(req.rid, None)
        storage.release_row(req)

    # ---- scheduler loop: poll the load, then insert or free ----

    def check_pending(self) -> None:
        if self._pending is not None:
            self._check_pending(self._pending)

    def abort(self) -> None:
        """Drop the in-flight prefetch: wait for its load to stop writing the
        slots, then free them without an insert. Called where every rank
        reaches the same point (memory release, shutdown)."""
        pending = self._pending
        if pending is None:
            return
        self._pending = None
        if pending.landed is None:
            # Blocks until every layer future resolved on this rank.
            self._linker.cache_linker.finish_layer_wise_loading(pending.counter_index)
        self._finish(pending, landed=False, error="Prefetch aborted.")

    def _check_pending(self, pending: _PendingPrefetch) -> None:
        """Poll the store read with one reduction per scheduler iteration.
        Every rank must take the same step in the same iteration: loads land
        at different times per rank and the insert is collective state, so
        the per-rank verdicts are reduced (MIN) before they are acted on."""
        linker = self._linker
        futures = linker.layer_done_counter.futures.get(pending.counter_index)
        if futures is None:
            # The counter was reset under the load: done, but nothing can
            # vouch for the slots.
            done, landed = 1, 0
        else:
            done = int(all(future.done() for future in futures))
            landed = int(done and all(future.exception() is None for future in futures))
        verdict = torch.tensor([done, landed], dtype=torch.int64)
        self._scheduler.tree_cache._all_reduce_attn_groups(
            verdict, torch.distributed.ReduceOp.MIN
        )
        done, landed = verdict.tolist()
        if not done:
            return
        # Non-blocking now that every layer resolved; drops the counter entry.
        linker.cache_linker.finish_layer_wise_loading(pending.counter_index)
        linker.adopt_load_timing(pending.counter_index, [pending.req.rid])
        pending.landed = bool(landed)
        self._pending = None
        self._finish(pending, pending.landed)

    def _all_ranks(self, local: bool) -> bool:
        flag = torch.tensor([int(local)], dtype=torch.int)
        self._scheduler.tree_cache._all_reduce_attn_groups(
            flag, torch.distributed.ReduceOp.MIN
        )
        return bool(flag.item())

    def _finish(
        self, pending: _PendingPrefetch, landed: bool, error: Optional[str] = None
    ) -> None:
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH
        from sglang.srt.mem_cache.common import release_kv_cache

        req = pending.req
        req.finished_reason = FINISH_LENGTH(0)
        if not landed:
            req.skip_radix_cache_insert = True
            self._linker.timings.pop(req.rid, None)
        # With checkpoint the tail [device_hit, prefix_len) joins the tree
        # like a finished request's KV; without it the row's tail is freed.
        release_kv_cache(req, self._scheduler.tree_cache, checkpoint=landed)
        elapsed_ms = self._elapsed_ms(pending.started_at)
        loaded_tokens = pending.loaded_tokens if landed else 0
        matched_tokens = pending.device_hit_tokens + loaded_tokens
        logger.info(
            "Prefetch %s: device_hit=%d loaded=%d elapsed_ms=%.1f",
            "landed" if landed else "failed",
            pending.device_hit_tokens,
            loaded_tokens,
            elapsed_ms,
        )
        collector = self._scheduler.metrics_collector
        if collector is not None:
            collector.observe_prefetch(loaded_tokens, elapsed_ms, landed)
        if not landed and error is None:
            error = "External linker load failed."
        self._send_output(
            PrefetchReqOutput(
                rid=pending.recv_req.rid,
                matched_tokens=matched_tokens,
                loaded_tokens=loaded_tokens,
                elapsed_ms=elapsed_ms,
                error=None if landed else error,
            ),
            pending.recv_req,
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.perf_counter() - started_at) * 1000.0

    def _failure(
        self,
        recv_req: PrefetchReqInput,
        started_at: float,
        error: str,
        *,
        matched_tokens: int = 0,
    ) -> PrefetchReqOutput:
        return PrefetchReqOutput(
            rid=recv_req.rid,
            matched_tokens=matched_tokens,
            loaded_tokens=0,
            elapsed_ms=self._elapsed_ms(started_at),
            error=error,
        )
