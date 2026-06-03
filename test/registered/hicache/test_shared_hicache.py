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
    SharedHiCacheConfig,
    SharedHiCacheTransferBackendType,
    normalize_shared_hicache_server_config,
)
from sglang.srt.mem_cache.shared_hicache.plan import SharedHiCachePlan
from sglang.srt.mem_cache.shared_hicache.scheduler_mixin import (
    SharedHiCacheSchedulerMixin,
    SharedHiCachePrepareStatus,
)
from sglang.srt.mem_cache.shared_hicache.service import (
    _decode_control_payload,
    _encode_control_payload,
)
from sglang.srt.mem_cache.shared_hicache.source import (
    ResolvedHostPage,
    execute_source_transfer_request,
    parse_source_transfer_request,
    resolve_host_pages,
)
from sglang.srt.mem_cache.shared_hicache.target import SharedHiCacheTarget
from sglang.srt.mem_cache.shared_hicache.transfer import (
    NixlSharedHiCacheTransferBackend,
)
from sglang.srt.mem_cache.utils import block_hash_aliases, hash_str_to_int64


def _make_plan(block_hashes, **overrides):
    plan = {
        "plan_id": "plan-1",
        "request_id": "request-1",
        "target_worker_id": "target-worker",
        "source_worker_id": "source-worker",
        "source_host": "127.0.0.1",
        "source_bootstrap_port": 39006,
        "source_medium": StorageMedium.CPU.value,
        "block_hashes": block_hashes,
        "planned_prefix_blocks": len(block_hashes),
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
        self.alloc_calls = []

    def alloc(self, need_size):
        self.alloc_calls.append(need_size)
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


class FakeHostPool:
    def __init__(self):
        self.pages = {}

    def get_data_page(self, index, flat=True):
        return self.pages[int(index)]


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
            mem_pool_host=FakeHostPool(),
            mem_pool_device_allocator=self.device_allocator,
        )
        self.token_to_kv_pool_allocator = self.device_allocator
        self.evict_count = 0
        self.evict_frees = 0
        self.evict_requests = []

    def is_chunk_cache(self):
        return False

    def evict(self, params):
        self.evict_count += 1
        self.evict_requests.append(params.num_tokens)
        if self.evict_frees > 0:
            current = self.device_allocator.available_size()
            self.device_allocator.max_alloc_size = current + self.evict_frees
        return SimpleNamespace(num_tokens_evicted=0)


class FakeTransferBackend:
    name = "nixl"
    enabled = True
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

    def transfer(self, handle):
        return self.initial_state

    def check_xfer_state(self, handle):
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

    def has_reuse_plan(self, req):
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

    def get_num_allocatable_reqs(self, running_bs):
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
    manager._set_parallel_metadata(
        {
            "tp_rank": 1,
            "tp_size": 2,
            "pp_rank": 0,
            "pp_size": 1,
            "attn_cp_rank": 0,
            "attn_cp_size": 1,
            "attn_tp_rank": 1,
            "attn_tp_size": 2,
            "attn_dp_rank": 0,
            "attn_dp_size": 1,
        }
    )
    return manager


class TestSharedHiCache(unittest.TestCase):
    def test_control_payload_uses_json_bytes(self):
        payload = {
            "kind": "shared_hicache_transfer_request",
            "transfer_id": "transfer-1",
            "target_page_indices": [1, 2],
        }

        encoded = _encode_control_payload(payload)

        self.assertIsInstance(encoded, bytes)
        self.assertNotIn(b"\x80", encoded[:1])
        self.assertEqual(_decode_control_payload([encoded]), payload)

    def test_server_config_uses_host_and_bootstrap_port(self):
        enabled, worker_id, config = normalize_shared_hicache_server_config(
            enable_shared_hicache=True,
            worker_id="target-worker",
            host="127.0.0.1",
            bootstrap_port=39000,
            timeout_secs=1.0,
            transfer_backend="nixl",
            enable_hierarchical_cache=True,
        )

        self.assertTrue(enabled)
        self.assertEqual(worker_id, "target-worker")
        self.assertIsInstance(config, SharedHiCacheConfig)
        self.assertEqual(config.bootstrap_port, 39000)
        self.assertEqual(
            config.transfer_backend,
            SharedHiCacheTransferBackendType.NIXL,
        )

        manager = SharedHiCacheManager.__new__(SharedHiCacheManager)
        manager._set_parallel_metadata({"tp_rank": 3, "tp_size": 4})
        self.assertEqual(
            manager._local_control_endpoint(config),
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
            normalize_shared_hicache_server_config(
                enable_shared_hicache=True,
                worker_id="target-worker",
                host="",
                bootstrap_port=39000,
                timeout_secs=1.0,
                transfer_backend="auto",
                enable_hierarchical_cache=True,
            )

    def test_plan_uses_canonical_schema_only(self):
        plan = SharedHiCachePlan.from_dict(
            _make_plan([11], source_tp_rank=0, source_tp_size=1)
        )

        self.assertEqual(plan.source_medium, StorageMedium.CPU.value)
        self.assertNotIn("source_endpoint", plan.to_dict())
        self.assertEqual(plan.source_host, "127.0.0.1")
        self.assertEqual(plan.source_bootstrap_port, 39006)
        self.assertEqual(plan.block_hashes, (11,))

        with self.assertRaisesRegex(ValueError, "source_endpoint is not supported"):
            SharedHiCachePlan.from_dict(
                _make_plan([11], source_endpoint="tcp://127.0.0.1:39007")
            )

        with self.assertRaisesRegex(ValueError, "block_hash must be an integer"):
            SharedHiCachePlan.from_dict(_make_plan([{"block_hash": 11}]))

    def test_generate_req_input_reads_shared_plan_from_cache_hints(self):
        req = GenerateReqInput(
            text="hello",
            sampling_params={},
            cache_hints={"shared_hicache": _make_plan([11])},
        )

        req.normalize_batch_and_arguments()

        self.assertIsInstance(req.shared_hicache_plan, SharedHiCachePlan)
        self.assertEqual(req.shared_hicache_plan.block_hashes, (11,))

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
        backend._capture_telemetry = False

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
        for alias in block_hash_aliases(block_hash):
            self.assertIn(alias, matches)
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
        cache._update_leaf_status = lambda node: None
        cache._update_host_leaf_status = lambda node: None
        cache.inc_lock_ref = lambda node: None
        cache.dec_lock_ref = lambda node: None
        cache.evict_host = lambda num_tokens: 0

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

    def test_hicache_host_index_does_not_claim_protected_node(self):
        node = TreeNode()
        node.host_value = torch.tensor([0, 1], dtype=torch.int64)
        node.hash_value = ["aa" * 32]
        index = HiCacheHostBlockIndex(page_size=2)
        index.index_node(node)
        node.protect_host()

        self.assertFalse(index.claim_unprotected_node_for_eviction(node))

        block_hash = hash_str_to_int64("aa" * 32)
        matches = index.lookup({block_hash})
        for alias in block_hash_aliases(block_hash):
            self.assertIn(alias, matches)
        node.release_host()

    def test_hicache_host_index_claim_drops_unprotected_node(self):
        node = TreeNode()
        node.host_value = torch.tensor([0, 1], dtype=torch.int64)
        node.hash_value = ["aa" * 32]
        index = HiCacheHostBlockIndex(page_size=2)
        index.index_node(node)

        self.assertTrue(index.claim_unprotected_node_for_eviction(node))

        block_hash = hash_str_to_int64("aa" * 32)
        self.assertEqual(index.lookup({block_hash}), {})

    def test_source_resolves_protected_hicache_host_pages(self):
        kv_hash = hash_str_to_int64("aa" * 32)
        identity_hash = 123
        node = TreeNode()
        node.host_value = torch.tensor([100, 102], dtype=torch.int64)
        tree = FakeTree(page_size=2)
        tree.cache_controller.mem_pool_host.pages[100] = torch.tensor(
            [1, 2, 3, 4], dtype=torch.uint8
        )

        def lookup(wanted_hashes, *, protect=False):
            self.assertEqual(set(wanted_hashes), {kv_hash})
            self.assertTrue(protect)
            node.protect_host()
            return {kv_hash: (node, 0, "aa" * 32)}, [node]

        tree.lookup_hicache_host_blocks = lookup
        plan = SharedHiCachePlan.from_dict(
            _make_plan([identity_hash], kv_block_hashes=[kv_hash])
        )

        pages, reason = resolve_host_pages(
            tree,
            plan,
            start_block=0,
            max_blocks=1,
            worker_id="source-worker",
        )

        self.assertEqual(reason, "ok")
        self.assertEqual(
            pages, [ResolvedHostPage(identity_hash, "aa" * 32, bytes([1, 2, 3, 4]))]
        )
        self.assertEqual(node.host_ref_counter, 0)

    def test_source_rejects_unprotected_hicache_host_lookup(self):
        kv_hash = hash_str_to_int64("aa" * 32)
        identity_hash = 123
        node = TreeNode()
        node.host_value = torch.tensor([100, 102], dtype=torch.int64)
        node.hash_value = ["aa" * 32]
        tree = FakeTree(page_size=2)

        def lookup(wanted_hashes, *, protect=False):
            self.assertEqual(set(wanted_hashes), {kv_hash})
            self.assertTrue(protect)
            return {kv_hash: (node, 0, "aa" * 32)}, []

        tree.lookup_hicache_host_blocks = lookup
        plan = SharedHiCachePlan.from_dict(
            _make_plan([identity_hash], kv_block_hashes=[kv_hash])
        )

        pages, reason = resolve_host_pages(
            tree,
            plan,
            start_block=0,
            max_blocks=1,
            worker_id="source-worker",
        )

        self.assertEqual(pages, [])
        self.assertEqual(reason, "unprotected_hicache_host_lookup")
        self.assertEqual(node.host_ref_counter, 0)

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
            tp_rank=0,
            tp_size=2,
            attn_tp_size=2,
        )

        self.assertFalse(response["ok"])
        self.assertIn("wrong_source_tp_rank_for_target", response["reason"])

    def test_target_direct_transfer_allocation_does_not_evict(self):
        tree = FakeTree()
        tree.device_allocator.fail_alloc = True
        target = SharedHiCacheTarget(tree_cache=tree, metrics_collector=None)

        self.assertIsNone(target.alloc_device_indices(4))
        self.assertEqual(tree.evict_count, 0)

    def test_manager_submits_partial_target_staging_when_full_alloc_fails(self):
        manager = _make_manager()
        manager.tree_cache.device_allocator.max_alloc_size = 4
        manager.direct_transfer = FakeTransferBackend()
        manager.target_cache = SharedHiCacheTarget(
            tree_cache=manager.tree_cache, metrics_collector=None
        )
        manager.metrics_collector = None
        manager.timeout_secs = 1.0
        manager.prefetch_stop_policy = "timeout"
        manager.endpoint = "tcp://127.0.0.1:39008"
        manager._target_transfer_capacity = None
        manager._pending_fetches = {}
        manager._finished_plan_keys = set()
        manager._finished_plan_prefix_lens = {}
        starts = []
        payloads = []
        manager.target_transfer_tracker = SimpleNamespace(
            start=lambda transfer_id: starts.append(transfer_id),
            finish=lambda transfer_id: None,
        )
        manager.source_service = SimpleNamespace(
            send=lambda endpoint, payload: payloads.append((endpoint, payload))
        )
        plan = SharedHiCachePlan.from_dict(
            _make_plan(
                [11, 22, 33],
                source_tp_size=2,
                target_tp_size=2,
            )
        )
        req = SimpleNamespace(
            rid="rid-1",
            shared_hicache_plan=plan,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            fill_ids=array("q", range(8)),
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            extra_key=None,
            last_node=None,
        )

        result = manager.prepare_reuse(req)

        self.assertTrue(result.pending)
        self.assertEqual(manager.tree_cache.evict_count, 1)
        self.assertEqual(manager.tree_cache.device_allocator.alloc_calls, [6, 4])
        self.assertEqual(len(payloads), 1)
        endpoint, payload = payloads[0]
        self.assertEqual(endpoint, "tcp://127.0.0.1:39007")
        self.assertEqual(payload["start_block"], 0)
        self.assertEqual(payload["max_blocks"], 2)
        self.assertEqual(payload["target_page_indices"], [100, 101])
        self.assertEqual(len(starts), 1)
        pending = manager._pending_fetches["rid-1"]
        self.assertEqual(pending.expected_hashes, (11, 22))
        self.assertEqual(pending.target_start_block, 0)
        self.assertEqual(pending.device_indices.numel(), 4)

    def test_manager_evicts_target_cache_before_partial_staging(self):
        manager = _make_manager()
        manager.tree_cache.device_allocator.max_alloc_size = 0
        manager.tree_cache.evict_frees = 6
        manager.direct_transfer = FakeTransferBackend()
        manager.target_cache = SharedHiCacheTarget(
            tree_cache=manager.tree_cache, metrics_collector=None
        )
        manager.metrics_collector = None
        manager.timeout_secs = 1.0
        manager.prefetch_stop_policy = "timeout"
        manager.endpoint = "tcp://127.0.0.1:39008"
        manager._target_transfer_capacity = None
        manager._pending_fetches = {}
        manager._finished_plan_keys = set()
        manager._finished_plan_prefix_lens = {}
        starts = []
        payloads = []
        manager.target_transfer_tracker = SimpleNamespace(
            start=lambda transfer_id: starts.append(transfer_id),
            finish=lambda transfer_id: None,
        )
        manager.source_service = SimpleNamespace(
            send=lambda endpoint, payload: payloads.append((endpoint, payload))
        )
        plan = SharedHiCachePlan.from_dict(
            _make_plan(
                [11, 22, 33],
                source_tp_size=2,
                target_tp_size=2,
            )
        )
        req = SimpleNamespace(
            rid="rid-1",
            shared_hicache_plan=plan,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            host_hit_length=0,
            fill_ids=array("q", range(8)),
            return_logprob=False,
            logprob_start_len=-1,
            positional_embed_overrides=None,
            extra_key=None,
            last_node=None,
        )

        result = manager.prepare_reuse(req)

        self.assertTrue(result.pending)
        self.assertEqual(manager.tree_cache.evict_count, 1)
        self.assertEqual(manager.tree_cache.evict_requests, [6])
        self.assertEqual(manager.tree_cache.device_allocator.alloc_calls, [6, 6])
        self.assertEqual(len(payloads), 1)
        endpoint, payload = payloads[0]
        self.assertEqual(endpoint, "tcp://127.0.0.1:39007")
        self.assertEqual(payload["start_block"], 0)
        self.assertEqual(payload["max_blocks"], 3)
        self.assertEqual(payload["target_page_indices"], [100, 101, 102])
        self.assertEqual(len(starts), 1)
        pending = manager._pending_fetches["rid-1"]
        self.assertEqual(pending.expected_hashes, (11, 22, 33))
        self.assertEqual(pending.target_start_block, 0)
        self.assertEqual(pending.device_indices.numel(), 6)

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

    def test_scheduler_tp_prepare_failure_falls_back_all_ranks(self):
        manager = FakeScheduleManager(prefix_len=8)
        scheduler = FakeConsensusScheduler(
            manager,
            status_overrides=[
                [SharedHiCachePrepareStatus.Ready],
                [SharedHiCachePrepareStatus.Failed],
            ],
        )
        req = FakeSharedHiCacheReq("rid-1", local_prefix_len=24)

        pending_rids = scheduler._prepare_shared_hicache_for_schedule_batch([req])

        self.assertEqual(pending_rids, set())
        self.assertEqual(
            scheduler.status_inputs,
            [[SharedHiCachePrepareStatus.Ready], [SharedHiCachePrepareStatus.Ready]],
        )
        self.assertEqual(scheduler.prefix_inputs, [[24], []])
        self.assertEqual(manager.prepared, ["rid-1"])
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
