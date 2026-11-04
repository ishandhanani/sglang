"""PrefetchReqInput handling in the scheduler with the external linker mocked:
the match reports N tokens in the store, the bulk load completes, and the
loaded tail is handed to the tree through the finished-request insert. Plus
the refusals, the memory gate, the hint socket and the tokenizer-side
sender."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import zmq

from sglang.srt.managers.io_struct import (
    FlushCacheReqInput,
    PrefetchReqInput,
    PrefetchReqOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.scheduler_components.prefetch_wrapper import (
    NOT_SERVED_ERROR,
    SchedulerPrefetchWrapper,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    LayerWiseLoadCounter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4
DEVICE_HIT = 2 * PAGE
STORE_HIT = 3 * PAGE
NUM_LAYERS = 3
TOKEN_IDS = list(range(100, 100 + DEVICE_HIT + STORE_HIT + 1))


class _FakeLinker:
    """The slice of UnifiedCacheLinkerWrapper + backend the prefetch touches."""

    def __init__(self):
        self.counter = LayerWiseLoadCounter(NUM_LAYERS)
        self.cache_linker = SimpleNamespace(
            start_bulk_loading=MagicMock(side_effect=self._start),
            finish_layer_wise_loading=MagicMock(side_effect=self.counter.finish),
            pending_broadcast=MagicMock(return_value=False),
            has_other_broadcasts=MagicMock(return_value=False),
            broadcast_bulk_layers=MagicMock(return_value=None),
            discard_broadcast=MagicMock(),
        )
        self.hit_rids = set()
        self.inflight_load_rids = []
        self.private_load_indices = {}
        self.timings = {}
        self.released = []
        self.refuse_load = False
        self.tail = torch.arange(1000, 1000 + STORE_HIT, dtype=torch.int64)

    @property
    def layer_done_counter(self):
        return self.counter

    def has_hit(self, rid):
        return rid in self.hit_rids

    def release_request(self, rid):
        self.released.append(rid)

    def adopt_load_timing(self, counter_index, rids):
        pass

    def mark_timing_kind(self, rid, kind):
        self.timing_kinds = getattr(self, "timing_kinds", {})
        self.timing_kinds[rid] = kind

    def load_back(self, req):
        # The real load_back queues the transfer, allocates private slots and
        # records them for prefix_device_indices.
        self.hit_rids.discard(req.rid)
        if self.refuse_load:
            return 0, req.last_node
        self.private_load_indices[req.rid] = self.tail
        self.inflight_load_rids.append(req.rid)
        return STORE_HIT, req.last_node

    def _start(self):
        return self.counter.update_producer()


class TestSchedulerPrefetch(unittest.TestCase):
    def _new_scheduler(self, *, store_hit=STORE_HIT, budget_tokens=1 << 20):
        linker = _FakeLinker()
        device_path = torch.arange(500, 500 + DEVICE_HIT, dtype=torch.int64)

        def prefix_device_indices(req):
            # The path to the match node (empty at the root) plus the private tail.
            path = device_path if req.last_node != 0 else device_path[:0]
            return torch.cat([path, linker.private_load_indices[req.rid]])[
                : req.prefix_len
            ]

        tree_cache = MagicMock()
        tree_cache.linker = linker
        tree_cache.components = {ComponentType.FULL: object()}
        tree_cache.prefix_device_indices.side_effect = prefix_device_indices
        tree_cache.lock.return_value = "lock"
        tree_cache.page_size = PAGE
        tree_cache.root_node_handle.return_value = 0
        tree_cache._all_reduce_attn_groups = MagicMock()  # single rank

        def match_kv_cache(req, tree_cache, token_ids, **kwargs):
            req.prefix_len = DEVICE_HIT
            req.last_node = 7
            req.host_hit_length = store_hit
            if store_hit:
                linker.hit_rids.add(req.rid)
            return SimpleNamespace(cache_protected_len=None)

        def acquire(reqs):
            reqs[0].kv.req_pool_idx = 3
            return [3]

        storage = SimpleNamespace(acquire=MagicMock(side_effect=acquire))
        storage.release_row = MagicMock()
        allocator = MagicMock()
        # The prefill budget the memory gate consults: free + evictable minus
        # the running batch's decode reserve.
        allocator.create_prefill_budget.return_value = SimpleNamespace(
            total_offset=0, remaining_total=budget_tokens
        )
        scheduler = SimpleNamespace(
            tree_cache=tree_cache,
            model_config=SimpleNamespace(vocab_size=5000, context_len=4096),
            req_to_token_pool=MagicMock(),
            token_to_kv_pool_allocator=allocator,
            running_batch=SimpleNamespace(is_empty=lambda: True, reqs=[]),
            new_token_ratio_tracker=SimpleNamespace(current=1.0),
            metrics_collector=None,
            linker=linker,
            storage=storage,
            match_kv_cache=MagicMock(side_effect=match_kv_cache),
            release_kv_cache=MagicMock(),
            send_output=MagicMock(),
            memory=SimpleNamespace(prefetch_max_tokens=None),
        )
        scheduler.prefetch_wrapper = SchedulerPrefetchWrapper(
            scheduler=scheduler, send_output=scheduler.send_output, served_dp_rank=0
        )
        return scheduler

    def _patches(self, scheduler):
        return [
            patch(
                "sglang.srt.mem_cache.common.match_kv_cache",
                scheduler.match_kv_cache,
            ),
            patch(
                "sglang.srt.mem_cache.static_kv_storage.kv_storage_of",
                return_value=scheduler.storage,
            ),
            patch(
                "sglang.srt.mem_cache.common.release_kv_cache",
                scheduler.release_kv_cache,
            ),
            patch(
                "sglang.srt.runtime_context.get_memory",
                return_value=scheduler.memory,
            ),
        ]

    def _handle(self, scheduler, token_ids=TOKEN_IDS, rid="rid-1", dp_rank=None):
        p = self._patches(scheduler)
        with p[0], p[1], p[2], p[3]:
            return scheduler.prefetch_wrapper.handle(
                PrefetchReqInput(rid=rid, token_ids=token_ids, dp_rank=dp_rank)
            )

    def _check_pending(self, scheduler):
        with self._patches(scheduler)[2]:
            scheduler.prefetch_wrapper.check_pending()

    def test_store_hit_loads_then_inserts(self):
        scheduler = self._new_scheduler()
        linker = scheduler.linker

        output = self._handle(scheduler)

        # Deferred: the load is in flight, nothing answered yet.
        self.assertIsNone(output)
        self.assertTrue(scheduler.prefetch_wrapper.has_pending())
        req = scheduler.prefetch_wrapper._pending.req
        self.assertEqual(list(req.origin_input_ids), TOKEN_IDS)
        self.assertEqual(linker.timing_kinds, {req.rid: "prefetch"})
        # Whole prefix matched: no `input_len - 1` cap.
        _, kwargs = scheduler.match_kv_cache.call_args
        self.assertIsNone(kwargs["max_prefix_len"])
        # The prefetch left the wrapper's batch bookkeeping, taking its slots
        # with it, and started its own load batch.
        self.assertEqual(linker.inflight_load_rids, [])
        self.assertNotIn(req.rid, linker.private_load_indices)
        linker.cache_linker.start_bulk_loading.assert_called_once()
        # The row holds the device prefix followed by the loaded tail.
        self.assertEqual(req.prefix_len, DEVICE_HIT + STORE_HIT)
        (row_slice, indices), _ = scheduler.req_to_token_pool.write.call_args
        self.assertEqual(row_slice, (3, slice(0, DEVICE_HIT + STORE_HIT)))
        self.assertEqual(
            indices.tolist(),
            list(range(500, 500 + DEVICE_HIT)) + list(range(1000, 1000 + STORE_HIT)),
        )
        self.assertEqual(req.kv.kv_committed_len, DEVICE_HIT + STORE_HIT)
        self.assertEqual(req.kv.cache_protected_len, DEVICE_HIT)
        self.assertEqual(req.lock, "lock")

        # Layers still loading: polling neither blocks nor answers.
        self._check_pending(scheduler)
        self.assertTrue(scheduler.prefetch_wrapper.has_pending())
        scheduler.release_kv_cache.assert_not_called()

        for layer in range(NUM_LAYERS):
            linker.counter.complete(0, layer)
        self._check_pending(scheduler)

        self.assertFalse(scheduler.prefetch_wrapper.has_pending())
        linker.cache_linker.finish_layer_wise_loading.assert_called_once_with(0)
        # The tail joins the tree through the finished-request insert.
        scheduler.release_kv_cache.assert_called_once()
        (released, tree_cache), kwargs = scheduler.release_kv_cache.call_args
        self.assertIs(released, req)
        self.assertIs(tree_cache, scheduler.tree_cache)
        self.assertTrue(kwargs["checkpoint"])
        self.assertTrue(req.finished())
        self.assertFalse(req.skip_radix_cache_insert)
        out, answered = scheduler.send_output.call_args.args
        self.assertIsInstance(out, PrefetchReqOutput)
        self.assertIsNone(out.error)
        self.assertEqual(out.rid, "rid-1")
        self.assertEqual(out.matched_tokens, DEVICE_HIT + STORE_HIT)
        self.assertEqual(out.loaded_tokens, STORE_HIT)
        self.assertGreater(out.elapsed_ms, 0.0)
        self.assertEqual(answered.token_ids, TOKEN_IDS)

    def test_failed_load_frees_without_insert(self):
        scheduler = self._new_scheduler()
        linker = scheduler.linker
        self._handle(scheduler)
        req = scheduler.prefetch_wrapper._pending.req

        linker.counter.complete(0, 0)
        linker.counter.fail(0, RuntimeError("store read failed"))
        self._check_pending(scheduler)

        self.assertFalse(scheduler.prefetch_wrapper.has_pending())
        _, kwargs = scheduler.release_kv_cache.call_args
        self.assertFalse(kwargs["checkpoint"])
        self.assertTrue(req.skip_radix_cache_insert)
        out = scheduler.send_output.call_args.args[0]
        self.assertIsNotNone(out.error)
        self.assertEqual(out.loaded_tokens, 0)
        self.assertEqual(out.matched_tokens, DEVICE_HIT)

    def test_no_store_hit_answers_immediately(self):
        scheduler = self._new_scheduler(store_hit=0)

        output = self._handle(scheduler)

        self.assertIsNone(output.error)
        self.assertEqual(output.matched_tokens, DEVICE_HIT)
        self.assertEqual(output.loaded_tokens, 0)
        self.assertFalse(scheduler.prefetch_wrapper.has_pending())
        scheduler.storage.acquire.assert_not_called()
        scheduler.linker.cache_linker.start_bulk_loading.assert_not_called()

    def test_refusals(self):
        scheduler = self._new_scheduler()
        self._handle(scheduler)

        output = self._handle(scheduler, rid="rid-2")
        self.assertIn("already in progress", output.error)
        self.assertEqual(output.rid, "rid-2")

        mamba = self._new_scheduler()
        mamba.tree_cache.components = {
            ComponentType.FULL: object(),
            ComponentType.MAMBA: object(),
        }
        self.assertIn("Mamba", self._handle(mamba).error)
        mamba.match_kv_cache.assert_not_called()

        swa = self._new_scheduler()
        swa.tree_cache.components = {
            ComponentType.FULL: object(),
            ComponentType.SWA: object(),
        }
        self.assertIn("SWA", self._handle(swa).error)

        scheduler.tree_cache.linker = None
        output = self._handle(scheduler, rid="rid-3", token_ids=[1])
        self.assertIn("not enabled", output.error)

    def test_bad_input_is_answered_not_raised(self):
        """A malformed hint (strings, bools, out-of-range ids, too long, the
        wrong dp_rank) is decided before any collective and answered with a
        PrefetchReqOutput carrying error; the match never runs."""
        cases = [
            (dict(token_ids=["1", "2"]), "list of ints"),
            (dict(token_ids=[]), "non-empty"),
            (dict(token_ids=[True, True]), "list of ints"),
            (dict(token_ids=[1, -5]), "outside the vocabulary"),
            (dict(token_ids=[1, 5000]), "outside the vocabulary"),
            (dict(token_ids=[1] * 4097), "prefetch limit"),
            (dict(token_ids=[1, 2], dp_rank=1), NOT_SERVED_ERROR),
            (dict(token_ids=[1, 2], dp_rank="0"), "dp_rank"),
        ]
        for kwargs, needle in cases:
            scheduler = self._new_scheduler()
            output = self._handle(scheduler, rid="bad", **kwargs)
            self.assertIsInstance(output, PrefetchReqOutput, kwargs)
            self.assertIn(needle, output.error, kwargs)
            self.assertEqual(output.rid, "bad")
            scheduler.match_kv_cache.assert_not_called()
            self.assertFalse(scheduler.prefetch_wrapper.has_pending())

        # --prefetch-max-tokens caps below the context length.
        scheduler = self._new_scheduler()
        scheduler.memory.prefetch_max_tokens = 8
        self.assertIn(
            "prefetch limit (8)", self._handle(scheduler, token_ids=[1] * 9).error
        )

        # A bug in the validation itself is answered too.
        scheduler = self._new_scheduler()
        scheduler.model_config = None
        scheduler.tree_cache.components = None
        self.assertIn("rejected", self._handle(scheduler).error)

    def test_failures_past_the_first_collective_propagate(self):
        """After validation the ranks take collectives together; a rank-local
        exception is not swallowed (that would leave the other ranks in the
        next all-reduce) but propagates like a request's would."""
        scheduler = self._new_scheduler()
        scheduler.match_kv_cache.side_effect = RuntimeError("store lookup timed out")
        with self.assertRaisesRegex(RuntimeError, "store lookup timed out"):
            self._handle(scheduler, rid="boom")

        scheduler = self._new_scheduler()
        self._handle(scheduler)
        scheduler.tree_cache._all_reduce_attn_groups.side_effect = RuntimeError(
            "nccl gone"
        )
        with self.assertRaisesRegex(RuntimeError, "nccl gone"):
            self._check_pending(scheduler)
        self.assertTrue(scheduler.prefetch_wrapper.has_pending())

    def test_memory_gate_truncates_to_the_budget(self):
        """The hit is matched only up to the KV headroom left after the running
        batch's decode reserve, page-aligned; no headroom refuses the hint."""
        scheduler = self._new_scheduler(budget_tokens=3 * PAGE + 1)
        self._handle(scheduler)
        req = scheduler.prefetch_wrapper._pending.req
        self.assertEqual(len(req.origin_input_ids), 3 * PAGE)
        scheduler.token_to_kv_pool_allocator.create_prefill_budget.assert_called_once_with(
            scheduler.tree_cache
        )

        # The running batch's decode reserve comes off the top.
        scheduler = self._new_scheduler(budget_tokens=3 * PAGE)
        scheduler.running_batch = SimpleNamespace(
            is_empty=lambda: False,
            reqs=[
                SimpleNamespace(
                    sampling_params=SimpleNamespace(
                        max_new_tokens=PAGE, ignore_eos=True
                    ),
                    output_ids=[],
                )
            ],
        )
        self._handle(scheduler)
        budget = scheduler.token_to_kv_pool_allocator.create_prefill_budget.return_value
        self.assertEqual(budget.total_offset, PAGE)

        scheduler = self._new_scheduler(budget_tokens=PAGE - 1)
        output = self._handle(scheduler)
        self.assertIn("No KV headroom", output.error)
        scheduler.match_kv_cache.assert_not_called()

    def test_row_shortage_and_refused_load_release_everything(self):
        scheduler = self._new_scheduler()
        scheduler.storage.acquire.side_effect = RuntimeError("no state slot")
        output = self._handle(scheduler)
        self.assertIn("No free request slot", output.error)
        self.assertEqual(scheduler.linker.released, ["prefetch-rid-1"])
        self.assertFalse(scheduler.prefetch_wrapper.has_pending())

        scheduler = self._new_scheduler()
        scheduler.linker.refuse_load = True
        output = self._handle(scheduler)
        self.assertIn("refused the load", output.error)
        self.assertEqual(output.matched_tokens, DEVICE_HIT + STORE_HIT)
        scheduler.tree_cache.unlock.assert_called_once_with("lock")
        scheduler.storage.release_row.assert_called_once()
        self.assertEqual(scheduler.linker.released, ["prefetch-rid-1"])
        self.assertNotIn("prefetch-rid-1", scheduler.linker.private_load_indices)

    def test_abort_waits_for_the_load_and_frees_without_insert(self):
        scheduler = self._new_scheduler()
        linker = scheduler.linker
        self._handle(scheduler)
        req = scheduler.prefetch_wrapper._pending.req
        for layer in range(NUM_LAYERS):
            linker.counter.complete(0, layer)

        with self._patches(scheduler)[2]:
            scheduler.prefetch_wrapper.abort()
            scheduler.prefetch_wrapper.abort()  # idempotent

        self.assertFalse(scheduler.prefetch_wrapper.has_pending())
        linker.cache_linker.finish_layer_wise_loading.assert_called_once_with(0)
        _, kwargs = scheduler.release_kv_cache.call_args
        self.assertFalse(kwargs["checkpoint"])
        self.assertTrue(req.skip_radix_cache_insert)
        out = scheduler.send_output.call_args.args[0]
        self.assertIn("aborted", out.error)

    def test_hint_socket_round_trip(self):
        """A DEALER (and a REQ) peer sends one msgpack frame on the hint
        socket; the request joins the recv list and the reply goes back to
        the same identity."""
        ctx = zmq.Context.instance()
        router = ctx.socket(zmq.ROUTER)
        router.setsockopt(zmq.LINGER, 0)
        endpoint = "inproc://hint-test"
        router.bind(endpoint)
        dealer = ctx.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect(endpoint)
        rq = ctx.socket(zmq.REQ)
        rq.setsockopt(zmq.LINGER, 0)
        rq.connect(endpoint)
        receiver = SchedulerRequestReceiver(
            recv_from_tokenizer=MagicMock(),
            recv_from_rpc=None,
            recv_skipper=None,
            input_blocker=None,
            mm_receiver=None,
            server_args=MagicMock(),
            model_config=MagicMock(),
            max_recv_per_poll=-1,
            stream_output=MagicMock(),
            get_last_batch=MagicMock(),
            rust_server_mode=False,
            hint_socket=router,
        )
        try:
            dealer.send(msgpack_encode(PrefetchReqInput(rid="d1", token_ids=[1, 2])))
            rq.send(msgpack_encode(PrefetchReqInput(rid="r1", token_ids=[3])))
            dealer.send(b"not msgpack")
            # Another struct of the shared IPC codec is not a hint: the strict
            # decoder rejects anything but a PrefetchReqInput.
            dealer.send(msgpack_encode(FlushCacheReqInput()))
            recv_reqs = []
            deadline = 50
            while len(recv_reqs) < 2 and deadline:
                receiver._pull_hint_reqs(recv_reqs)  # one hint per poll
                deadline -= 1
            self.assertEqual(sorted(r.rid for r in recv_reqs), ["d1", "r1"])
            self.assertTrue(all(isinstance(r, PrefetchReqInput) for r in recv_reqs))
            self.assertEqual(set(receiver.hint_routes), {"d1", "r1"})
            # REQ peers carry the empty delimiter, DEALER peers do not.
            self.assertEqual(len(receiver.hint_routes["r1"]), 2)
            self.assertEqual(len(receiver.hint_routes["d1"]), 1)

            # A second hint with an in-flight rid is answered at once.
            dealer.send(msgpack_encode(PrefetchReqInput(rid="d1", token_ids=[9])))
            for _ in range(50):
                receiver._pull_hint_reqs(recv_reqs)
                if dealer.poll(10):
                    break
            dup = msgpack_decode(dealer.recv())
            self.assertEqual((dup.rid, dup.loaded_tokens), ("d1", 0))
            self.assertIn("in flight", dup.error)
            self.assertEqual(len(recv_reqs), 2)

            self.assertTrue(
                receiver.reply_hint(PrefetchReqOutput(rid="d1", loaded_tokens=8))
            )
            self.assertTrue(
                receiver.reply_hint(PrefetchReqOutput(rid="r1", error="nope"))
            )
            self.assertFalse(receiver.reply_hint(PrefetchReqOutput(rid="other")))
            out = msgpack_decode(dealer.recv())
            self.assertEqual((out.rid, out.loaded_tokens), ("d1", 8))
            out = msgpack_decode(rq.recv())
            self.assertEqual((out.rid, out.error), ("r1", "nope"))
            self.assertEqual(receiver.hint_routes, {})
            receiver.close()
            self.assertTrue(router.closed)
        finally:
            dealer.close()
            rq.close()
            router.close()

    def test_ipc_round_trip(self):
        """The type rides the scheduler's input socket as a tagged msgpack array."""
        req = PrefetchReqInput(rid="abc", token_ids=[5, 6, 7], dp_rank=2)
        decoded = msgpack_decode(msgpack_encode(req))
        self.assertIsInstance(decoded, PrefetchReqInput)
        self.assertEqual(
            (decoded.rid, decoded.token_ids, decoded.dp_rank), ("abc", [5, 6, 7], 2)
        )
        self.assertIsNone(PrefetchReqInput(rid="x", token_ids=[1]).dp_rank)

        out = PrefetchReqOutput(rid="abc", matched_tokens=8, loaded_tokens=4)
        decoded = msgpack_decode(msgpack_encode(out))
        self.assertIsInstance(decoded, PrefetchReqOutput)
        self.assertEqual(decoded.loaded_tokens, 4)
        self.assertIsNone(decoded.error)

    def test_tokenizer_manager_prefetch(self):
        """TokenizerManager.prefetch fans the request out to every engine and
        returns the reply of the one serving dp_rank; a stale reply with
        another rid is ignored."""
        sent = []

        async def communicator(obj):
            sent.append(obj)
            return [
                PrefetchReqOutput(rid="stale", loaded_tokens=99),
                PrefetchReqOutput(rid=obj.rid, error=f"{NOT_SERVED_ERROR} 1; no"),
                PrefetchReqOutput(rid=obj.rid, matched_tokens=16, loaded_tokens=16),
            ]

        manager = SimpleNamespace(
            auto_create_handle_loop=MagicMock(), prefetch_communicator=communicator
        )
        out = asyncio.run(TokenizerControlMixin.prefetch(manager, [1, 2, 3], dp_rank=1))
        self.assertEqual((sent[0].token_ids, sent[0].dp_rank), ([1, 2, 3], 1))
        self.assertEqual(out.rid, sent[0].rid)
        self.assertEqual(out.loaded_tokens, 16)

        async def only_mismatch(obj):
            return [PrefetchReqOutput(rid=obj.rid, error=f"{NOT_SERVED_ERROR} 3; no")]

        manager.prefetch_communicator = only_mismatch
        out = asyncio.run(TokenizerControlMixin.prefetch(manager, [1], dp_rank=3))
        self.assertIn(NOT_SERVED_ERROR, out.error)


if __name__ == "__main__":
    unittest.main()
