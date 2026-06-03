import time
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.hicache_host_index import HiCacheHostBlockIndex
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.shared_hicache.manager import SharedHiCacheManager
from sglang.srt.mem_cache.shared_hicache.config import (
    normalize_shared_hicache_server_config,
)
from sglang.srt.mem_cache.shared_hicache.plan import SharedHiCachePlan
from sglang.srt.mem_cache.shared_hicache.scheduler_mixin import (
    SharedHiCacheSchedulerMixin,
    SharedHiCachePrepareStatus,
)
from sglang.srt.mem_cache.shared_hicache.source import (
    execute_source_transfer_request,
    parse_source_transfer_request,
)
from sglang.srt.mem_cache.shared_hicache.topology import SharedHiCacheTopology
from sglang.srt.mem_cache.shared_hicache.transfer.nixl import (
    NixlSharedHiCacheTransferBackend,
)
from sglang.srt.mem_cache.utils import hash_str_to_int64


def _make_plan(router_block_hashes, **overrides):
    plan = {
        "plan_id": "plan-1",
        "request_id": "request-1",
        "target_worker_id": "target-worker",
        "source_worker_id": "source-worker",
        "source_host": "127.0.0.1",
        "source_bootstrap_port": 39006,
        "source_medium": StorageMedium.CPU.value,
        "router_block_hashes": router_block_hashes,
        "engine_block_hashes": router_block_hashes,
        "planned_prefix_blocks": len(router_block_hashes),
        "block_size_tokens": 2,
        "created_at_ms": 1,
        "expires_at_ms": int(time.time() * 1000) + 60_000,
    }
    plan.update(overrides)
    return plan


class FakeDeviceAllocator:
    def __init__(self):
        self.fail_alloc = False
        self.max_alloc_size = None

    def alloc(self, need_size):
        if self.fail_alloc:
            return None
        if self.max_alloc_size is not None and need_size > self.max_alloc_size:
            return None
        return torch.arange(200, 200 + need_size)

    def free(self, indices):
        return len(indices)

    def available_size(self):
        if self.fail_alloc:
            return 0
        if self.max_alloc_size is not None:
            return self.max_alloc_size
        return 1_000_000


class FakeDoneEvent:
    def query(self):
        return True

    def synchronize(self):
        pass


class FakeWriteThroughController:
    write_policy = "write_through"

    def __init__(self):
        self.ack_write_queue = []
        self.next_host_index = 1000

    def write(self, *, device_indices, node_id, **kwargs):
        assert kwargs == {}
        host_indices = torch.arange(
            self.next_host_index,
            self.next_host_index + len(device_indices),
            dtype=torch.int64,
        )
        self.next_host_index += len(device_indices)
        self.ack_write_queue.append((None, FakeDoneEvent(), [node_id]))
        return host_indices


class FakeTree:
    def __init__(self, page_size=2):
        self.page_size = page_size
        self.root_node = TreeNode()
        self.root_node.key = RadixKey(array("q"))
        self.device_allocator = FakeDeviceAllocator()
        self.cache_controller = SimpleNamespace(
            mem_pool_device_allocator=self.device_allocator,
        )
        self.token_to_kv_pool_allocator = self.device_allocator

    def is_chunk_cache(self):
        return False

    def evict(self, _params):
        return SimpleNamespace(num_tokens_evicted=0)


class FakeTransferBackend:
    name = "nixl"
    target_session_id = "target-session"
    target_kv_ptrs = [1]
    target_kv_item_lens = [64]

    def target_descriptor(self):
        return {"backend": self.name, "session_id": self.target_session_id}


class FakeNixlAgent:
    def __init__(self, *, initial_state="DONE", check_error=None):
        self.initial_state = initial_state
        self.check_error = check_error
        self.released = []

    def transfer(self, _handle):
        return self.initial_state

    def check_xfer_state(self, _handle):
        if self.check_error is not None:
            raise self.check_error
        return "DONE"

    def release_xfer_handle(self, handle):
        self.released.append(handle)


class FakeSharedHiCacheReq:
    def __init__(self, rid, *, local_prefix_len=0):
        self.rid = rid
        self.shared_hicache_plan = True
        self.shared_hicache_max_prefix_len = None
        self.local_prefix_len = local_prefix_len
        self.host_hit_length = 0
        self.prefix_indices = torch.arange(local_prefix_len, dtype=torch.int64)
        self.init_calls = []

    def init_next_round_input(self, tree_cache=None, cow_mamba=None):
        assert tree_cache is not None
        assert cow_mamba in (None, False)
        self.init_calls.append(self.shared_hicache_max_prefix_len)
        prefix_len = self.local_prefix_len
        if self.shared_hicache_max_prefix_len is not None:
            prefix_len = min(prefix_len, self.shared_hicache_max_prefix_len)
        self.prefix_indices = torch.arange(prefix_len, dtype=torch.int64)


class FakeScheduleManager:
    def __init__(self, prefix_len, *, pending=False):
        self.prefix_len = prefix_len
        self.pending = pending
        self.prepared = []
        self.released = []

    def has_reuse_plan(self, _req):
        return True

    def prepare_reuse(self, req):
        self.prepared.append(req.rid)
        return SimpleNamespace(pending=self.pending, prefix_len=self.prefix_len)

    def release_request(self, rid):
        self.released.append(rid)


class FakeScheduler(SharedHiCacheSchedulerMixin):
    def __init__(self, manager):
        self.shared_hicache_manager = manager
        self.ps = SimpleNamespace(tp_size=1)
        self.tree_cache = object()
        self.server_args = SimpleNamespace(prefill_max_requests=None)
        self.chunked_req = None
        self.enable_priority_preemption = False

    def get_num_allocatable_reqs(self, _running_bs):
        return 1


class FakeConsensusScheduler(FakeScheduler):
    def __init__(self, manager, *, status_overrides=None, prefix_overrides=None):
        super().__init__(manager)
        self.status_overrides = list(status_overrides or [])
        self.prefix_overrides = list(prefix_overrides or [])
        self.status_inputs = []
        self.prefix_inputs = []

    def _sync_shared_hicache_status_min_batch(self, values):
        self.status_inputs.append(list(values))
        if self.status_overrides:
            return list(self.status_overrides.pop(0))
        return list(values)

    def _sync_shared_hicache_int_min_batch(self, values):
        self.prefix_inputs.append(list(values))
        if self.prefix_overrides:
            return list(self.prefix_overrides.pop(0))
        return list(values)


def _make_manager():
    manager = SharedHiCacheManager.__new__(SharedHiCacheManager)
    manager.worker_id = "target-worker"
    manager.tree_cache = FakeTree(page_size=2)
    manager._set_topology(
        SharedHiCacheTopology(
            tp_rank=1,
            tp_size=2,
            attn_tp_rank=1,
            attn_tp_size=2,
        )
    )
    return manager


class TestSharedHiCache(unittest.TestCase):
    def test_server_config_uses_host_and_bootstrap_port(self):
        enabled, worker_id, bootstrap_port, transfer_backend = (
            normalize_shared_hicache_server_config(
                enable_shared_hicache=True,
                worker_id="target-worker",
                bootstrap_port=39000,
                transfer_backend="nixl",
                enable_hierarchical_cache=True,
            )
        )

        self.assertTrue(enabled)
        self.assertEqual(worker_id, "target-worker")
        self.assertEqual(bootstrap_port, 39000)
        self.assertEqual(transfer_backend, "nixl")

        manager = SharedHiCacheManager.__new__(SharedHiCacheManager)
        manager._set_topology(SharedHiCacheTopology(tp_rank=3, tp_size=4))
        self.assertEqual(
            manager._local_control_endpoint(
                SimpleNamespace(
                    host="127.0.0.1",
                    shared_hicache_bootstrap_port=bootstrap_port,
                )
            ),
            "tcp://127.0.0.1:39003",
        )
        self.assertEqual(
            manager._source_control_endpoint_for_plan(
                SharedHiCachePlan.from_dict(
                    _make_plan([11], source_tp_rank=3, source_tp_size=4)
                )
            ),
            "tcp://127.0.0.1:39009",
        )

        with self.assertRaisesRegex(ValueError, "host"):
            manager._local_control_endpoint(
                SimpleNamespace(
                    host="",
                    shared_hicache_bootstrap_port=bootstrap_port,
                )
            )

        with self.assertRaisesRegex(
            ValueError,
            "--enable-shared-hicache requires --shared-hicache-transfer-backend nixl",
        ):
            normalize_shared_hicache_server_config(
                enable_shared_hicache=True,
                worker_id="target-worker",
                bootstrap_port=39000,
                transfer_backend=None,
                enable_hierarchical_cache=True,
            )

        with self.assertRaisesRegex(ValueError, "shared_hicache_transfer_backend"):
            normalize_shared_hicache_server_config(
                enable_shared_hicache=True,
                worker_id="target-worker",
                bootstrap_port=39000,
                transfer_backend="auto",
                enable_hierarchical_cache=True,
            )

    def test_plan_uses_canonical_schema_only(self):
        plan = SharedHiCachePlan.from_dict(
            _make_plan([11], source_tp_rank=0, source_tp_size=1)
        )

        self.assertEqual(plan.source_medium, StorageMedium.CPU.value)
        self.assertEqual(plan.source_host, "127.0.0.1")
        self.assertEqual(plan.source_bootstrap_port, 39006)
        self.assertEqual(plan.router_block_hashes, (11,))
        self.assertEqual(plan.engine_block_hashes, (11,))

        with self.assertRaisesRegex(ValueError, "router_block_hashes"):
            SharedHiCachePlan.from_dict(_make_plan([{"block_hash": 11}]))

        missing_engine_hashes = _make_plan([11])
        missing_engine_hashes.pop("engine_block_hashes")
        with self.assertRaisesRegex(ValueError, "missing engine_block_hashes"):
            SharedHiCachePlan.from_dict(missing_engine_hashes)

        with self.assertRaisesRegex(ValueError, "engine_block_hashes length"):
            SharedHiCachePlan.from_dict(
                _make_plan([11], engine_block_hashes=[11, 12])
            )

    def test_plan_keeps_router_and_engine_hash_representations_separate(self):
        signed_hash = hash_str_to_int64("aa" * 32)
        unsigned_hash = signed_hash & (2**64 - 1)

        plan = SharedHiCachePlan.from_dict(
            _make_plan([unsigned_hash], engine_block_hashes=[signed_hash])
        )

        self.assertGreater(unsigned_hash, 2**63 - 1)
        self.assertEqual(plan.router_block_hashes, (unsigned_hash,))
        self.assertEqual(plan.engine_block_hashes, (signed_hash,))

    def test_generate_req_input_reads_shared_plan_from_cache_hints(self):
        req = GenerateReqInput(
            text="hello",
            sampling_params={},
            cache_hints={"shared_hicache": _make_plan([11])},
        )

        req.normalize_batch_and_arguments()

        self.assertIsInstance(req.shared_hicache_plan, SharedHiCachePlan)
        self.assertEqual(req.shared_hicache_plan.router_block_hashes, (11,))

    def test_generate_req_input_normalizes_batched_cache_hints(self):
        req = GenerateReqInput(
            text=["hello", "world"],
            sampling_params=[{}, {}],
            cache_hints=[
                {"shared_hicache": _make_plan([11])},
                None,
            ],
        )

        req.normalize_batch_and_arguments()

        self.assertIsInstance(req.shared_hicache_plan[0], SharedHiCachePlan)
        self.assertIsNone(req.shared_hicache_plan[1])
        self.assertIsInstance(req[0].shared_hicache_plan, SharedHiCachePlan)
        self.assertIsNone(req[1].shared_hicache_plan)

        with self.assertRaisesRegex(ValueError, "cache_hints must be a list"):
            GenerateReqInput(
                text=["hello", "world"],
                sampling_params=[{}, {}],
                cache_hints={"shared_hicache": _make_plan([11])},
            ).normalize_batch_and_arguments()

        with self.assertRaisesRegex(
            ValueError, "cache_hints.shared_hicache does not support"
        ):
            GenerateReqInput(
                text="hello",
                sampling_params={"n": 2},
                cache_hints={"shared_hicache": _make_plan([11])},
            ).normalize_batch_and_arguments()

    def test_nixl_transfer_handle_released_on_failure(self):
        backend = NixlSharedHiCacheTransferBackend.__new__(
            NixlSharedHiCacheTransferBackend
        )

        err_agent = FakeNixlAgent(initial_state="ERR")
        with self.assertRaisesRegex(RuntimeError, "NIXL direct KV transfer failed"):
            backend._wait_for_transfer(err_agent, "err-handle")
        self.assertEqual(err_agent.released, ["err-handle"])

        stalled_agent = FakeNixlAgent(
            initial_state="WAITING",
            check_error=RuntimeError("poll failed"),
        )
        with self.assertRaisesRegex(RuntimeError, "poll failed"):
            backend._wait_for_transfer(stalled_agent, "poll-handle")
        self.assertEqual(stalled_agent.released, ["poll-handle"])

    def test_hiradix_cpu_events_maintain_host_index(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 2
        cache.enable_kv_cache_events = False
        cache.hicache_host_index = HiCacheHostBlockIndex(cache.page_size)
        node = TreeNode()
        node.host_value = torch.tensor([0, 1], dtype=torch.int64)
        node.hash_value = ["aa" * 32]

        cache._record_store_event(node, medium=StorageMedium.CPU)

        block_hash = hash_str_to_int64("aa" * 32)
        matches, protected = cache.lookup_hicache_host_blocks(
            {block_hash}, protect=True
        )
        self.assertIn(block_hash, matches)
        self.assertEqual(protected, [node])
        self.assertEqual(node.host_ref_counter, 1)
        node.release_host()

        cache._record_remove_event(node, medium=StorageMedium.CPU)
        self.assertEqual(cache.lookup_hicache_host_blocks({block_hash}), {})

    def test_hiradix_split_pending_write_through_publishes_cpu_prefix(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.disable = False
        cache.page_size = 2
        cache.is_eagle = False
        cache.enable_storage = False
        cache.enable_kv_cache_events = True
        cache.enable_shared_hicache = True
        cache.hicache_host_index = None
        cache.cache_controller = FakeWriteThroughController()
        cache.write_through_threshold = 1
        cache.ongoing_write_through = {}
        cache.kv_event_queue = []
        cache.evictable_size_ = 0
        cache.protected_size_ = 0
        cache.evictable_leaves = set()
        cache.evictable_host_leaves = set()
        cache.root_node = TreeNode(priority=-(10**9))
        cache.root_node.key = RadixKey(array("q"))
        cache.root_node.value = []
        cache.root_node.host_value = []
        cache.root_node.hash_value = []
        cache.root_node.lock_ref = 1
        def ignore_node(node):
            self.assertIsNotNone(node)

        def evict_host(num_tokens):
            self.assertGreaterEqual(num_tokens, 0)
            return 0

        cache._update_leaf_status = ignore_node
        cache._update_host_leaf_status = ignore_node
        cache.inc_lock_ref = ignore_node
        cache.dec_lock_ref = ignore_node
        cache.evict_host = evict_host

        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 3, 4])),
                value=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            )
        )
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", [1, 2, 5, 6])),
                value=torch.tensor([20, 21, 22, 23], dtype=torch.int64),
            )
        )

        self.assertEqual(
            [
                tuple(event.token_ids)
                for event in cache.kv_event_queue
                if event.medium == StorageMedium.CPU
            ],
            [],
        )

        cache._drain_finished_write_through_acks(
            cache._count_finished_write_through_acks()
        )

        cpu_events = [
            event for event in cache.kv_event_queue if event.medium == StorageMedium.CPU
        ]
        prefix_hash = cpu_events[0].block_hashes[0]
        self.assertEqual(
            [(event.parent_block_hash, tuple(event.token_ids)) for event in cpu_events],
            [
                (None, (1, 2)),
                (prefix_hash, (3, 4)),
                (prefix_hash, (5, 6)),
            ],
        )

    def test_source_transfer_rejects_wrong_tp_rank_metadata(self):
        plan = SharedHiCachePlan.from_dict(
            _make_plan(
                [11],
                source_tp_rank=0,
                source_tp_size=2,
                target_tp_rank=1,
                target_tp_size=2,
            )
        )
        request, error = parse_source_transfer_request(
            payload={
                "transfer_id": "transfer-1",
                "target_control_endpoint": "tcp://127.0.0.1:49999",
                "plan": plan.to_dict(),
                "start_block": 0,
                "max_blocks": 1,
                "target_session_id": "target-session",
                "transfer_backend": "nixl",
                "target_metadata": {
                    "backend": "nixl",
                    "session_id": "target-session",
                    "tp_rank": 1,
                    "tp_size": 2,
                },
                "target_kv_ptrs": [1],
                "target_kv_item_lens": [64],
                "target_page_indices": [0],
            },
            transfer_backend=FakeTransferBackend(),
            tree_cache=FakeTree(),
        )
        self.assertIsNone(error)

        response = execute_source_transfer_request(
            request=request,
            transfer_backend=FakeTransferBackend(),
            tree_cache=FakeTree(),
            worker_id="source-worker",
            topology=SharedHiCacheTopology(tp_rank=0, tp_size=2, attn_tp_size=2),
        )

        self.assertFalse(response["ok"])
        self.assertIn("wrong_source_tp_rank_for_target", response["reason"])

    def test_shared_hicache_device_insert_does_not_write_through(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        captured = []

        def insert(params):
            captured.append(params)
            return SimpleNamespace(prefix_len=0)

        cache.insert = insert
        key = RadixKey(array("q", [1, 2]))
        value = torch.tensor([10, 11], dtype=torch.int64)

        result = cache.insert_shared_hicache_device_blocks(key=key, value=value)

        self.assertEqual(result.prefix_len, 0)
        self.assertIsInstance(captured[0], InsertParams)
        self.assertTrue(captured[0].chunked)

    def test_manager_validates_target_tp_rank(self):
        manager = _make_manager()
        wrong_rank = SharedHiCachePlan.from_dict(
            _make_plan(
                [11],
                source_tp_rank=0,
                source_tp_size=2,
                target_tp_rank=0,
                target_tp_size=2,
            )
        )
        rank_generic = SharedHiCachePlan.from_dict(
            _make_plan([11], source_tp_size=2, target_tp_size=2)
        )

        self.assertEqual(
            manager._validate_plan(wrong_rank),
            "wrong_target_tp_rank:plan=0:local=1",
        )
        self.assertIsNone(manager._validate_plan(rank_generic))

    def test_scheduler_keeps_longer_local_prefix(self):
        manager = FakeScheduleManager(prefix_len=8)
        scheduler = FakeScheduler(manager)
        req = FakeSharedHiCacheReq("rid-1", local_prefix_len=24)

        pending_rids = scheduler._prepare_shared_hicache_for_schedule_batch([req])

        self.assertEqual(pending_rids, set())
        self.assertEqual(manager.prepared, ["rid-1"])
        self.assertEqual(req.shared_hicache_max_prefix_len, 24)
        self.assertEqual(req.init_calls, [None, 24])

    def test_scheduler_tp_probe_failure_falls_back_all_ranks(self):
        manager = FakeScheduleManager(prefix_len=8)
        scheduler = FakeConsensusScheduler(
            manager,
            status_overrides=[
                [SharedHiCachePrepareStatus.Failed],
            ],
        )
        req = FakeSharedHiCacheReq("rid-1", local_prefix_len=24)

        pending_rids = scheduler._prepare_shared_hicache_for_schedule_batch([req])

        self.assertEqual(pending_rids, set())
        self.assertEqual(scheduler.status_inputs, [[SharedHiCachePrepareStatus.Ready]])
        self.assertEqual(scheduler.prefix_inputs, [])
        self.assertEqual(manager.prepared, [])
        self.assertEqual(manager.released, ["rid-1"])
        self.assertIsNone(req.shared_hicache_plan)

    def test_scheduler_tp_pending_status_dominates_ready(self):
        manager = FakeScheduleManager(prefix_len=8)
        scheduler = FakeConsensusScheduler(
            manager,
            status_overrides=[
                [SharedHiCachePrepareStatus.Ready],
                [SharedHiCachePrepareStatus.Pending],
            ],
        )
        req = FakeSharedHiCacheReq("rid-1", local_prefix_len=24)

        pending_rids = scheduler._prepare_shared_hicache_for_schedule_batch([req])

        self.assertEqual(pending_rids, {"rid-1"})
        self.assertEqual(
            scheduler.status_inputs,
            [[SharedHiCachePrepareStatus.Ready], [SharedHiCachePrepareStatus.Ready]],
        )
        self.assertEqual(scheduler.prefix_inputs, [[24], []])
        self.assertEqual(manager.prepared, ["rid-1"])
        self.assertEqual(manager.released, [])


if __name__ == "__main__":
    unittest.main()
