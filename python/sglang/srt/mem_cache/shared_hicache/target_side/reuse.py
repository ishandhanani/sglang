from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.mem_cache.shared_hicache.control import (
    SHARED_HICACHE_TRANSFER_REQUEST,
    SharedHiCacheTargetTransferTracker,
    SharedHiCacheTransferHandle,
)
from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_DIRECT_TIMEOUT_REASON,
    SHARED_HICACHE_PLAN_VERSION,
    SHARED_HICACHE_SOURCE_MEDIUM,
    SharedHiCachePlan,
)
from sglang.srt.mem_cache.shared_hicache.route import shared_hicache_source_endpoint
from sglang.srt.mem_cache.shared_hicache.target_side.cache import SharedHiCacheTarget
from sglang.srt.mem_cache.shared_hicache.target_side.pending import (
    SharedHiCachePendingFetch,
)
from sglang.srt.mem_cache.shared_hicache.topology import SharedHiCacheTopology
from sglang.srt.mem_cache.shared_hicache.transfer.common import (
    SharedHiCacheTransferBackend,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedHiCacheResult:
    staged_tokens: int = 0
    prefix_len: int = 0
    pending: bool = False


@dataclass(frozen=True)
class SharedHiCacheDirectSubmitResult:
    transfer: Optional[SharedHiCacheTransferHandle] = None
    device_indices: Optional[torch.Tensor] = None
    reason: Optional[str] = None
    submitted_blocks: int = 0
    submitted_tokens: int = 0
    available_tokens_before: Optional[int] = None


def validate_shared_hicache_plan(
    plan: SharedHiCachePlan,
    *,
    worker_id: str,
    topology: SharedHiCacheTopology,
    page_size: int,
) -> Optional[str]:
    if plan.target_worker_id != worker_id:
        return "wrong_target_worker"
    rank_rejection = topology.validate_target_rank(plan)
    if rank_rejection is not None:
        return rank_rejection
    if plan.source_worker_id == plan.target_worker_id:
        return "source_is_target"
    if plan.plan_version != SHARED_HICACHE_PLAN_VERSION:
        return "unsupported_plan_version"
    if plan.is_expired():
        return "plan_expired"
    if plan.source_medium != SHARED_HICACHE_SOURCE_MEDIUM:
        return "unsupported_source_medium"
    if plan.block_size_tokens != page_size:
        return "incompatible_block_size"
    return None


class SharedHiCacheTargetReuse:
    def __init__(
        self,
        *,
        tree_cache,
        worker_id: str,
        topology: SharedHiCacheTopology,
        direct_transfer: SharedHiCacheTransferBackend,
        target_cache: SharedHiCacheTarget,
        target_transfer_tracker: SharedHiCacheTargetTransferTracker,
        endpoint: str,
        send_control_message: Callable[[str, Mapping[str, Any]], None],
        timeout_secs: float,
        prefetch_stop_policy: str,
        fetch_worker_limit: int,
        target_reserve_tokens: int,
        metrics_collector=None,
    ):
        self.tree_cache = tree_cache
        self.worker_id = worker_id
        self.topology = topology
        self.direct_transfer = direct_transfer
        self.target_cache = target_cache
        self.target_transfer_tracker = target_transfer_tracker
        self.endpoint = endpoint
        self._send_control_message = send_control_message
        self.timeout_secs = timeout_secs
        self.prefetch_stop_policy = prefetch_stop_policy
        self.target_reserve_tokens = max(0, int(target_reserve_tokens))
        self.metrics_collector = metrics_collector
        self._target_transfer_capacity = threading.BoundedSemaphore(
            max(1, int(fetch_worker_limit))
        )
        self._pending_fetches: dict[str, SharedHiCachePendingFetch] = {}
        self._finished_plan_keys: set[tuple[str, str]] = set()
        self._finished_plan_prefix_lens: dict[tuple[str, str], int] = {}

    def _observe_reuse(
        self,
        *,
        backend: str,
        outcome: str,
        reason: str,
        tokens: int = 0,
        wait_ms: Optional[float] = None,
        insert_ms: Optional[float] = None,
        transfer_bytes: Optional[int] = None,
    ) -> None:
        if self.metrics_collector is None:
            return
        self.metrics_collector.observe_shared_hicache(
            backend=backend,
            outcome=outcome,
            reason=reason,
            tokens=max(0, int(tokens)),
            wait_ms=wait_ms,
            insert_ms=insert_ms,
            transfer_bytes=transfer_bytes,
        )

    def _observe_staging(
        self,
        *,
        backend: str,
        outcome: str,
        reason: str,
        tokens: int,
    ) -> None:
        if self.metrics_collector is None:
            return
        self.metrics_collector.observe_shared_hicache_staging(
            backend=backend,
            outcome=outcome,
            reason=reason,
            tokens=max(0, int(tokens)),
        )

    def _try_acquire_fetch_worker(self) -> bool:
        return self._target_transfer_capacity.acquire(blocking=False)

    def _release_fetch_worker(self) -> None:
        try:
            self._target_transfer_capacity.release()
        except ValueError:
            logger.debug(
                "SharedHiCache fetch worker semaphore release ignored", exc_info=True
            )

    def _max_cacheable_blocks(self, req: Req) -> int:
        max_prefix_len = max(len(req.full_untruncated_fill_ids) - 1, 0)
        if req.return_logprob and req.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, req.logprob_start_len)
        if req.positional_embed_overrides is not None:
            max_prefix_len = 0
        return max_prefix_len // self.tree_cache.page_size

    def _validate_plan(self, plan: SharedHiCachePlan) -> Optional[str]:
        return validate_shared_hicache_plan(
            plan,
            worker_id=self.worker_id,
            topology=self.topology,
            page_size=self.tree_cache.page_size,
        )

    def _plan_key(self, req: Req, plan: SharedHiCachePlan) -> tuple[str, str]:
        return str(req.rid), plan.plan_id

    def reuse_plan_rejection(self, req: Req) -> Optional[str]:
        plan = getattr(req, "shared_hicache_plan", None)
        if plan is None:
            return None
        if not isinstance(plan, SharedHiCachePlan):
            return "invalid_plan"
        return self._validate_plan(plan)

    def observe_plan_skip(self, req: Req, reason: str) -> None:
        plan = getattr(req, "shared_hicache_plan", None)
        if plan is None:
            return

        backend = self.direct_transfer.name
        plan_id = "unknown"
        source_worker = "unknown"
        if isinstance(plan, SharedHiCachePlan):
            plan_id = plan.plan_id
            source_worker = plan.source_worker_id
            plan_key = self._plan_key(req, plan)
            if plan_key in self._finished_plan_keys:
                return
            self._finished_plan_keys.add(plan_key)

        logger.debug(
            "Skipping shared HiCache plan rid=%s plan_id=%s reason=%s source_worker=%s",
            req.rid,
            plan_id,
            reason,
            source_worker,
        )
        self._observe_reuse(
            backend=backend,
            outcome="skip",
            reason=reason,
        )

    def _pending_wait_ms(self, pending: SharedHiCachePendingFetch) -> Optional[float]:
        if pending.submitted_at <= 0:
            return None
        return (time.perf_counter() - pending.submitted_at) * 1000

    def _pending_ready_wait_ms(
        self, pending: SharedHiCachePendingFetch
    ) -> Optional[float]:
        done_at = pending.done_at or pending.transfer.done_at
        if done_at <= 0:
            return None
        return max(0.0, (time.perf_counter() - done_at) * 1000)

    def _pending_should_stop_waiting(
        self, pending: SharedHiCachePendingFetch
    ) -> tuple[bool, str]:
        policy = str(self.prefetch_stop_policy)
        if policy == "best_effort":
            return True, "best_effort_incomplete"
        if policy == "wait_complete":
            return False, ""
        if policy == "timeout":
            elapsed = time.perf_counter() - pending.submitted_at
            if self.timeout_secs >= 0 and elapsed > self.timeout_secs:
                return True, "prefetch_timeout"
            return False, ""
        return True, "unknown_prefetch_policy"

    def _pending_transfer_bytes(
        self, pending: SharedHiCachePendingFetch, page_count: int
    ) -> int:
        bytes_per_page = int(pending.bytes_per_page or 0)
        return int(page_count) * bytes_per_page if bytes_per_page > 0 else 0

    def has_pending(self) -> bool:
        return bool(self._pending_fetches)

    def _lock_request_prefix(self, req: Req) -> Optional[TreeNode]:
        last_node = getattr(req, "last_node", None)
        if last_node is None or last_node is self.tree_cache.root_node:
            return None
        self.tree_cache.inc_lock_ref(last_node)
        return last_node

    def _unlock_pending_prefix(self, pending: SharedHiCachePendingFetch) -> None:
        locked_node = pending.locked_node
        if locked_node is None:
            return
        pending.locked_node = None
        self.tree_cache.dec_lock_ref(locked_node)

    def _release_pending_fetch(self, pending: SharedHiCachePendingFetch) -> None:
        self._unlock_pending_prefix(pending)
        if pending.device_indices is None:
            self.target_transfer_tracker.finish(pending.transfer.transfer_id)
            self._release_fetch_worker()
            return
        backend = pending.backend
        transfer = pending.transfer
        if transfer.done():
            _, reason = transfer.result()
            self.target_transfer_tracker.finish(transfer.transfer_id)
            if str(reason).startswith(SHARED_HICACHE_DIRECT_TIMEOUT_REASON):
                self.target_cache.quarantine_device_indices(
                    pending.device_indices, reason, backend=backend
                )
            else:
                self.target_cache.free_device_indices(pending.device_indices)
        else:
            self.target_transfer_tracker.finish(transfer.transfer_id)
            self.target_cache.quarantine_device_indices(
                pending.device_indices,
                SHARED_HICACHE_DIRECT_TIMEOUT_REASON,
                backend=backend,
            )
        self._release_fetch_worker()

    def release_all_pending(self) -> None:
        for pending in self._pending_fetches.values():
            self._release_pending_fetch(pending)
        self._pending_fetches.clear()

    def _submit_direct_transfer(
        self,
        req: Req,
        plan: SharedHiCachePlan,
        *,
        start_block: int,
        token_count: int,
    ) -> SharedHiCacheDirectSubmitResult:
        direct_transfer = self.direct_transfer
        source_tp_rank = (
            int(plan.source_tp_rank)
            if plan.source_tp_rank is not None
            else int(self.topology.tp_rank)
        )
        source_control_endpoint = shared_hicache_source_endpoint(
            getattr(req, "shared_hicache_source_routes", ()),
            plan.source_worker_id,
            source_tp_rank,
        )
        if not source_control_endpoint:
            logger.warning(
                "Shared HiCache source route unavailable plan_id=%s source_worker=%s source_tp_rank=%s",
                plan.plan_id,
                plan.source_worker_id,
                source_tp_rank,
            )
            return SharedHiCacheDirectSubmitResult(
                reason="source_control_endpoint_unavailable"
            )
        if not self._try_acquire_fetch_worker():
            return SharedHiCacheDirectSubmitResult(reason="fetch_worker_unavailable")

        allocation = self.target_cache.alloc_page_aligned_device_indices(
            token_count,
            page_size=self.tree_cache.page_size,
            reserve_tokens=self.target_reserve_tokens,
        )
        device_indices = allocation.device_indices
        if device_indices is None:
            self._release_fetch_worker()
            return SharedHiCacheDirectSubmitResult(
                reason="target_staging_alloc_failed",
                available_tokens_before=allocation.available_tokens_before,
            )

        submitted_tokens = int(device_indices.numel())
        submitted_blocks = submitted_tokens // self.tree_cache.page_size
        if submitted_blocks <= 0:
            self.target_cache.free_device_indices(device_indices)
            self._release_fetch_worker()
            return SharedHiCacheDirectSubmitResult(
                reason="target_staging_alloc_failed",
                available_tokens_before=allocation.available_tokens_before,
            )

        target_page_indices = self.target_cache.device_indices_to_page_indices(
            device_indices
        )
        if target_page_indices is None:
            logger.warning(
                "Shared HiCache direct transfer got non page-aligned target device allocation"
            )
            self.target_cache.free_device_indices(device_indices)
            self._release_fetch_worker()
            return SharedHiCacheDirectSubmitResult(
                reason="target_page_alignment_failed",
                submitted_blocks=submitted_blocks,
                submitted_tokens=submitted_tokens,
                available_tokens_before=allocation.available_tokens_before,
            )
        transfer_id = uuid.uuid4().hex
        handle = SharedHiCacheTransferHandle(
            transfer_backend=direct_transfer,
            transfer_id=transfer_id,
            plan=plan,
            start_block=start_block,
            max_blocks=submitted_blocks,
            timeout_secs=self.timeout_secs,
            pop_source_completion=self.target_transfer_tracker.pop_completion,
        )
        target_descriptor = direct_transfer.target_descriptor()
        target_page_indices_payload = [int(index) for index in target_page_indices]
        self.target_transfer_tracker.start(transfer_id)
        try:
            self._send_control_message(
                source_control_endpoint,
                {
                    "kind": SHARED_HICACHE_TRANSFER_REQUEST,
                    "transfer_id": transfer_id,
                    "target_control_endpoint": self.endpoint,
                    "plan": plan.to_dict(),
                    "start_block": start_block,
                    "max_blocks": submitted_blocks,
                    "target_session_id": direct_transfer.target_session_id,
                    "transfer_backend": direct_transfer.name,
                    "target_metadata": target_descriptor,
                    "target_kv_ptrs": direct_transfer.target_kv_ptrs,
                    "target_kv_item_lens": direct_transfer.target_kv_item_lens,
                    "target_page_indices": target_page_indices_payload,
                },
            )
        except Exception:
            self.target_transfer_tracker.finish(transfer_id)
            self.target_cache.free_device_indices(device_indices)
            self._release_fetch_worker()
            logger.warning(
                "Shared HiCache direct transfer control send failed plan_id=%s source_worker=%s endpoint=%s",
                plan.plan_id,
                plan.source_worker_id,
                source_control_endpoint,
                exc_info=True,
            )
            return SharedHiCacheDirectSubmitResult(
                reason="control_send_failed",
                submitted_blocks=submitted_blocks,
                submitted_tokens=submitted_tokens,
                available_tokens_before=allocation.available_tokens_before,
            )
        return SharedHiCacheDirectSubmitResult(
            transfer=handle,
            device_indices=device_indices,
            submitted_blocks=submitted_blocks,
            submitted_tokens=submitted_tokens,
            available_tokens_before=allocation.available_tokens_before,
        )

    def has_reuse_plan(self, req: Req) -> bool:
        plan = getattr(req, "shared_hicache_plan", None)
        if not isinstance(plan, SharedHiCachePlan):
            return False
        return self.reuse_plan_rejection(req) is None

    def release_request(self, rid: str) -> None:
        rid = str(rid)
        pending = self._pending_fetches.pop(rid, None)

        if pending is not None:
            self._finished_plan_keys.add((rid, pending.plan.plan_id))
            self._observe_reuse(
                backend=pending.backend,
                outcome="miss",
                reason="request_released_pending",
                wait_ms=self._pending_wait_ms(pending),
            )
            self._release_pending_fetch(pending)

        self._finished_plan_keys = {
            key for key in self._finished_plan_keys if key[0] != rid
        }
        self._finished_plan_prefix_lens = {
            key: prefix_len
            for key, prefix_len in self._finished_plan_prefix_lens.items()
            if key[0] != rid
        }

    def prepare_reuse(self, req: Req) -> SharedHiCacheResult:
        backend = self.direct_transfer.name
        plan = getattr(req, "shared_hicache_plan", None)
        if plan is None:
            return SharedHiCacheResult()
        if not isinstance(plan, SharedHiCachePlan):
            logger.debug(
                "Ignoring invalid shared HiCache plan for rid=%s: expected SharedHiCachePlan got %s",
                req.rid,
                type(plan).__name__,
            )
            self._observe_reuse(
                backend=backend,
                outcome="skip",
                reason="invalid_plan",
            )
            return SharedHiCacheResult()

        rejection = self._validate_plan(plan)
        if rejection is not None:
            logger.debug(
                "Ignoring shared HiCache plan rid=%s plan_id=%s reason=%s",
                req.rid,
                plan.plan_id,
                rejection,
            )
            self._observe_reuse(
                backend=backend,
                outcome="skip",
                reason=rejection,
            )
            return SharedHiCacheResult()

        plan_key = self._plan_key(req, plan)
        if plan_key in self._finished_plan_keys:
            return SharedHiCacheResult(
                prefix_len=self._finished_plan_prefix_lens.get(plan_key, 0)
            )

        page_size = self.tree_cache.page_size
        matched_tokens = len(req.prefix_indices) + req.host_hit_length
        if matched_tokens % page_size != 0:
            logger.debug(
                "Skipping shared HiCache plan rid=%s plan_id=%s reason=unaligned_matched_tokens matched_tokens=%d page_size=%d",
                req.rid,
                plan.plan_id,
                matched_tokens,
                page_size,
            )
            self._observe_reuse(
                backend=backend,
                outcome="skip",
                reason="unaligned_matched_tokens",
            )
            return SharedHiCacheResult()
        computed_blocks = matched_tokens // page_size
        if computed_blocks < plan.start_block_index:
            logger.debug(
                "Skipping shared HiCache plan rid=%s plan_id=%s reason=before_plan_start computed_blocks=%d start_block_index=%d",
                req.rid,
                plan.plan_id,
                computed_blocks,
                plan.start_block_index,
            )
            self._observe_reuse(
                backend=backend,
                outcome="skip",
                reason="before_plan_start",
            )
            return SharedHiCacheResult()

        max_plan_blocks = max(
            self._max_cacheable_blocks(req) - plan.start_block_index, 0
        )
        planned_blocks = min(plan.planned_prefix_blocks, max_plan_blocks)
        plan_offset = computed_blocks - plan.start_block_index
        if planned_blocks <= plan_offset:
            logger.debug(
                "Skipping shared HiCache plan rid=%s plan_id=%s reason=no_remaining_planned_blocks planned_blocks=%d plan_offset=%d max_plan_blocks=%d",
                req.rid,
                plan.plan_id,
                planned_blocks,
                plan_offset,
                max_plan_blocks,
            )
            self._observe_reuse(
                backend=backend,
                outcome="skip",
                reason="no_remaining_planned_blocks",
            )
            self._finished_plan_keys.add(plan_key)
            self._finished_plan_prefix_lens[plan_key] = matched_tokens
            return SharedHiCacheResult(prefix_len=matched_tokens)

        pending = self._pending_fetches.get(str(req.rid))
        if pending is not None:
            if pending.plan.plan_id != plan.plan_id:
                self._pending_fetches.pop(str(req.rid), None)
                self._release_pending_fetch(pending)
            elif pending.transfer.poll() not in (KVPoll.Success, KVPoll.Failed):
                stop_waiting, reason = self._pending_should_stop_waiting(pending)
                if stop_waiting:
                    self._pending_fetches.pop(str(req.rid), None)
                    self._release_pending_fetch(pending)
                    self._finished_plan_keys.add(self._plan_key(req, pending.plan))
                    self._observe_reuse(
                        backend=pending.backend,
                        outcome="miss",
                        reason=reason,
                        wait_ms=self._pending_wait_ms(pending),
                    )
                    return SharedHiCacheResult()
                return SharedHiCacheResult(pending=True)
            else:
                return self._finish_pending_fetch(req, pending)

        logger.debug(
            "Submitting shared HiCache fetch rid=%s plan_id=%s source_worker=%s start_block=%d max_blocks=%d matched_tokens=%d",
            req.rid,
            plan.plan_id,
            plan.source_worker_id,
            plan_offset,
            planned_blocks - plan_offset,
            matched_tokens,
        )
        max_blocks = planned_blocks - plan_offset
        token_count = max_blocks * page_size
        expected_hashes = plan.planned_router_block_hashes[plan_offset:planned_blocks]
        self._observe_staging(
            backend=backend,
            outcome="planned",
            reason="remote_suffix_requested",
            tokens=token_count,
        )
        locked_node = self._lock_request_prefix(req)
        try:
            direct_submit = self._submit_direct_transfer(
                req,
                plan,
                start_block=plan_offset,
                token_count=token_count,
            )
        except Exception:
            if locked_node is not None:
                self.tree_cache.dec_lock_ref(locked_node)
            self._finished_plan_keys.add(plan_key)
            self._observe_staging(
                backend=backend,
                outcome="failed",
                reason="direct_submit_exception",
                tokens=token_count,
            )
            self._observe_reuse(
                backend=backend,
                outcome="miss",
                reason="direct_submit_exception",
            )
            logger.exception(
                "Shared HiCache direct transfer submit failed; falling back to local prefill "
                "rid=%s plan_id=%s source_worker=%s start_block=%d max_blocks=%d "
                "token_count=%d",
                req.rid,
                plan.plan_id,
                plan.source_worker_id,
                plan_offset,
                max_blocks,
                token_count,
            )
            return SharedHiCacheResult()
        transfer = direct_submit.transfer
        device_indices = direct_submit.device_indices
        direct_submit_reason = direct_submit.reason
        available_tokens_before = direct_submit.available_tokens_before
        submitted_blocks = int(direct_submit.submitted_blocks)
        submitted_tokens = int(direct_submit.submitted_tokens)
        if transfer is None:
            if locked_node is not None:
                self.tree_cache.dec_lock_ref(locked_node)
            self._finished_plan_keys.add(plan_key)
            direct_submit_reason = direct_submit_reason or "direct_submit_unavailable"
            if direct_submit_reason == "target_staging_alloc_failed":
                self._observe_staging(
                    backend=backend,
                    outcome="failed",
                    reason="target_staging_alloc_failed",
                    tokens=token_count,
                )
            logger.info(
                "Shared HiCache direct submit unavailable "
                "rid=%s plan_id=%s reason=%s source_worker=%s start_block=%d "
                "max_blocks=%d token_count=%d host_hit_length=%d "
                "available_tokens_before=%s",
                req.rid,
                plan.plan_id,
                direct_submit_reason,
                plan.source_worker_id,
                plan_offset,
                max_blocks,
                token_count,
                req.host_hit_length,
                available_tokens_before,
            )
            self._observe_reuse(
                backend=backend,
                outcome="miss",
                reason=direct_submit_reason,
            )
            return SharedHiCacheResult()
        if submitted_blocks < max_blocks:
            shortfall_tokens = token_count - submitted_tokens
            logger.info(
                "Shared HiCache target staging partially granted "
                "rid=%s plan_id=%s source_worker=%s start_block=%d "
                "requested_blocks=%d granted_blocks=%d requested_tokens=%d "
                "granted_tokens=%d failed_tokens=%d available_tokens_before=%s",
                req.rid,
                plan.plan_id,
                plan.source_worker_id,
                plan_offset,
                max_blocks,
                submitted_blocks,
                token_count,
                submitted_tokens,
                shortfall_tokens,
                available_tokens_before,
            )
            self._observe_staging(
                backend=backend,
                outcome="failed",
                reason="target_staging_alloc_shortfall",
                tokens=shortfall_tokens,
            )
        self._observe_staging(
            backend=backend,
            outcome="granted",
            reason="target_staging_alloc_granted",
            tokens=submitted_tokens,
        )
        max_blocks = submitted_blocks
        token_count = submitted_tokens
        expected_hashes = expected_hashes[:submitted_blocks]
        backend = "none"
        bytes_per_page = 0
        if device_indices is not None:
            backend = self.direct_transfer.name
            bytes_per_page = sum(self.direct_transfer.target_kv_item_lens)
        pending = SharedHiCachePendingFetch(
            plan=plan,
            plan_offset=plan_offset,
            target_start_block=plan.start_block_index + plan_offset,
            expected_hashes=expected_hashes,
            transfer=transfer,
            device_indices=device_indices,
            locked_node=locked_node,
            backend=backend,
            bytes_per_page=bytes_per_page,
            submitted_at=time.perf_counter(),
        )
        self._pending_fetches[str(req.rid)] = pending
        return SharedHiCacheResult(pending=True)

    def _finish_terminal_pending_fetch(
        self,
        req: Req,
        pending: SharedHiCachePendingFetch,
        *,
        outcome: str,
        reason: str,
        device_indices_action: str,
        transfer_page_count: Optional[int] = None,
        insert_ms: Optional[float] = None,
    ) -> SharedHiCacheResult:
        self._unlock_pending_prefix(pending)
        if pending.device_indices is not None:
            if device_indices_action == "free":
                self.target_cache.free_device_indices(pending.device_indices)
            elif device_indices_action == "quarantine":
                self.target_cache.quarantine_device_indices(
                    pending.device_indices,
                    reason,
                    backend=pending.backend,
                )
            elif device_indices_action != "keep":
                raise ValueError(
                    f"unknown SharedHiCache device action {device_indices_action}"
                )

        self._finished_plan_keys.add(self._plan_key(req, pending.plan))
        transfer_bytes = (
            None
            if transfer_page_count is None
            else self._pending_transfer_bytes(pending, transfer_page_count)
        )
        self._observe_reuse(
            backend=pending.backend,
            outcome=outcome,
            reason=reason,
            wait_ms=self._pending_wait_ms(pending),
            insert_ms=insert_ms,
            transfer_bytes=transfer_bytes,
        )
        self._release_fetch_worker()
        return SharedHiCacheResult()

    def _finish_pending_fetch(
        self, req: Req, pending: SharedHiCachePendingFetch
    ) -> SharedHiCacheResult:
        self._pending_fetches.pop(str(req.rid), None)
        plan = pending.plan
        if pending.done_at <= 0:
            pending.done_at = pending.transfer.done_at or time.perf_counter()

        try:
            pages, reason = pending.transfer.result()
        except Exception:
            self.target_transfer_tracker.finish(pending.transfer.transfer_id)
            logger.exception(
                "Shared HiCache fetch failed rid=%s plan_id=%s", req.rid, plan.plan_id
            )
            return self._finish_terminal_pending_fetch(
                req,
                pending,
                outcome="error",
                reason="fetch_exception",
                device_indices_action="free",
            )
        self.target_transfer_tracker.finish(pending.transfer.transfer_id)

        if not pages:
            logger.debug(
                "Shared HiCache source returned no pages rid=%s plan_id=%s reason=%s",
                req.rid,
                plan.plan_id,
                reason,
            )
            indeterminate_transfer = str(reason).startswith(
                SHARED_HICACHE_DIRECT_TIMEOUT_REASON
            )
            return self._finish_terminal_pending_fetch(
                req,
                pending,
                outcome="error" if indeterminate_transfer else "miss",
                reason=reason,
                device_indices_action="quarantine"
                if indeterminate_transfer
                else "free",
            )

        if len(pages) > len(pending.expected_hashes):
            logger.warning(
                "Shared HiCache source returned too many pages rid=%s plan_id=%s pages=%d expected=%d",
                req.rid,
                plan.plan_id,
                len(pages),
                len(pending.expected_hashes),
            )
            return self._finish_terminal_pending_fetch(
                req,
                pending,
                outcome="error",
                reason="too_many_pages",
                device_indices_action="free",
                transfer_page_count=len(pages),
            )

        expected_hashes = pending.expected_hashes[: len(pages)]
        if tuple(page.block_hash for page in pages) != expected_hashes:
            logger.warning(
                "Shared HiCache source returned non-contiguous pages rid=%s plan_id=%s",
                req.rid,
                plan.plan_id,
            )
            return self._finish_terminal_pending_fetch(
                req,
                pending,
                outcome="error",
                reason="non_contiguous_pages",
                device_indices_action="free",
                transfer_page_count=len(pages),
            )

        insert_start = time.perf_counter()
        try:
            if pending.device_indices is None:
                logger.warning(
                    "Shared HiCache direct transfer completed without target device indices"
                )
                return self._finish_terminal_pending_fetch(
                    req,
                    pending,
                    outcome="error",
                    reason="missing_target_device_indices",
                    device_indices_action="keep",
                )
            staged_tokens = self.target_cache.insert_device_pages(
                req,
                pages,
                device_indices=pending.device_indices,
                start_block=pending.target_start_block,
            )
        except Exception:
            insert_ms = (time.perf_counter() - insert_start) * 1000
            logger.exception(
                "Shared HiCache insert failed rid=%s plan_id=%s",
                req.rid,
                plan.plan_id,
            )
            return self._finish_terminal_pending_fetch(
                req,
                pending,
                outcome="error",
                reason="insert_exception",
                device_indices_action="keep",
                insert_ms=insert_ms,
                transfer_page_count=len(pages),
            )
        finally:
            self._unlock_pending_prefix(pending)
        insert_ms = (time.perf_counter() - insert_start) * 1000
        fetched_tokens = len(pages) * self.tree_cache.page_size
        prefix_len = (
            pending.target_start_block * self.tree_cache.page_size + fetched_tokens
        )
        if staged_tokens > 0:
            req.shared_hicache_hit_length = (
                getattr(req, "shared_hicache_hit_length", 0) + staged_tokens
            )
            wait_ms = self._pending_wait_ms(pending)
            ready_wait_ms = self._pending_ready_wait_ms(pending)
            logger.info(
                "Shared HiCache staged %d tokens rid=%s plan_id=%s source_worker=%s fetched_tokens=%d prefix_len=%d wait_ms=%s ready_wait_ms=%s insert_ms=%.3f direct=%s",
                staged_tokens,
                req.rid,
                plan.plan_id,
                plan.source_worker_id,
                fetched_tokens,
                prefix_len,
                "n/a" if wait_ms is None else f"{wait_ms:.3f}",
                "n/a" if ready_wait_ms is None else f"{ready_wait_ms:.3f}",
                insert_ms,
                pending.device_indices is not None,
            )
        self._finished_plan_keys.add(self._plan_key(req, plan))
        if staged_tokens > 0:
            self._finished_plan_prefix_lens[self._plan_key(req, plan)] = prefix_len
        outcome = "hit" if staged_tokens > 0 else "miss"
        self._observe_reuse(
            backend=pending.backend,
            outcome=outcome,
            reason=reason if staged_tokens > 0 else "insert_returned_zero",
            tokens=staged_tokens,
            wait_ms=self._pending_wait_ms(pending),
            insert_ms=insert_ms,
            transfer_bytes=self._pending_transfer_bytes(pending, len(pages)),
        )
        self._release_fetch_worker()
        return SharedHiCacheResult(staged_tokens=staged_tokens, prefix_len=prefix_len)
