"""Hybrid geometry and progressive delivery regressions (no GPU required)."""

import collections
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

pytest.importorskip("kvcr")

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
    _build_hybrid_mamba_device_pool_group,
    _dsv4_low_ratio_device_entries,
)
from sglang.srt.mem_cache.storage.kvcr.kvcr_config import KVCRLinkerConfig
from sglang.srt.mem_cache.storage.kvcr.kvcr_direct_linker import (
    KVCRDirectLinker,
    LayerWiseLoadCounter,
    _LoadBatch,
    _LoadPool,
)
from sglang.srt.mem_cache.storage.kvcr.kvcr_layout import plan_capacity
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_sparse_checkpoint_capacity_uses_physical_object_counts():
    layouts = {
        "kv": SimpleNamespace(object_bytes=8, subpools=("kv/0",), span_sizes=(8,)),
        "mamba": SimpleNamespace(object_bytes=32, subpools=("m/0",), span_sizes=(32,)),
    }
    dense = plan_capacity(layouts, 128)
    sparse = plan_capacity(layouts, 128, capacity_divisors={"mamba": 4})
    assert dense.pool_capacities == {"kv": 3, "mamba": 3}
    assert sparse.pool_capacities == {"kv": 8, "mamba": 2}
    assert sparse.total_bytes == 128
    # Crossing a checkpoint interval requires an entire additional state object.
    assert (
        plan_capacity(layouts, 135, capacity_divisors={"mamba": 4}).page_capacity == 8
    )
    for divisor in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            plan_capacity(layouts, 128, capacity_divisors={"mamba": divisor})
    with pytest.raises(ValueError, match="Unknown"):
        plan_capacity(layouts, 128, capacity_divisors={"missing": 4})


def test_low_ratio_indexer_rows_cover_one_logical_page():
    kv = SimpleNamespace(kv_buffer=[torch.zeros(4, 8, dtype=torch.uint8)])
    index = SimpleNamespace(
        page_size=2,
        index_k_with_scale_buffer=[torch.zeros(8, 3, dtype=torch.uint8)],
    )
    cache = SimpleNamespace(
        sources_by_ratio={2: [10]},
        kv_pools={2: kv},
        index_pools={2: index},
        start_layer=8,
    )
    entries = _dsv4_low_ratio_device_entries(cache, page_size=8)
    assert [entry.name for entry in entries] == [
        PoolName.DEEPSEEK_V4_C2,
        PoolName.DEEPSEEK_V4_C2_INDEXER,
    ]
    assert entries[1].layer_mapping == {2: 0}
    pointers, sizes = entries[1].get_page_buffer_meta(torch.arange(8, 16))
    assert pointers == [index.index_k_with_scale_buffer[0][2].data_ptr()]
    assert sizes == [6]
    assert _dsv4_low_ratio_device_entries(SimpleNamespace(), 8) == []


def test_mamba_entries_preserve_stage_layers_and_sibling_state():
    full = DevicePoolEntry(
        name=PoolName.KV,
        indices_from_pool=PoolName.KV,
        device_pool=None,
        components=[[torch.zeros(8, 2)]],
        layer_mapping={0: 0},
        page_size=2,
        rows_are_pages=False,
    )
    states = [torch.zeros(8, 3), torch.zeros(8, 7), torch.zeros(8, 1)]
    pool = SimpleNamespace(
        _iter_transfer_state_entries=lambda: iter(
            [
                ("conv", states[0], 0, 9),
                ("temporal", states[1], 0, 9),
                ("sibling", states[2], None, 10),
            ]
        )
    )
    cache = SimpleNamespace(
        full_kv_pool=None, full_attention_layer_id_mapping={8: 0}, start_layer=8
    )
    req = SimpleNamespace(mamba_pool=pool, translate_mamba_indices=lambda x: x)
    with patch(
        "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler._build_plain_kv_device_pool_group",
        return_value=DevicePoolGroup([full], 1, 2),
    ):
        group = _build_hybrid_mamba_device_pool_group(cache, req, 2)
    assert group.num_layers == 3
    assert not group.rank_replicated
    assert group.entries[0].layer_mapping == {0: 0}
    assert group.entries[1].layer_mapping == {1: [0, 1], 2: [2]}
    assert group.entries[1].get_prepared_layer_range_meta([2], 0) is None
    pointers, _, _ = group.entries[1].get_prepared_layer_range_meta([2], 2)
    assert pointers == [[states[2][2].data_ptr()]]


def test_mamba_external_transfer_owns_only_its_checkpoint_on_abort():
    component = MambaComponent.__new__(MambaComponent)
    component._alloc_mamba_slot = lambda: torch.tensor([7])
    component._free_mamba_value = Mock()
    transfer = component.build_external_linker_transfer(
        LinkerTransferPhase.LOAD, None, ["a", "b"]
    )
    assert transfer.keys == ["b"]
    req = SimpleNamespace(
        kv=SimpleNamespace(
            holds_mamba=True, mamba_pool_idx=torch.tensor(3), mamba_cow_src_index=None
        )
    )
    component.update_external_linker_load(
        ExternalLinkerLoadPhase.ABORT, req, None, transfer, 4
    )
    component._free_mamba_value.assert_called_once()
    assert req.kv.mamba_pool_idx.item() == 3


def test_mamba_commit_does_not_deliver_into_freed_duplicate_checkpoint():
    component = MambaComponent.__new__(MambaComponent)
    canonical = torch.tensor([9])
    node = SimpleNamespace(
        component_data={ComponentType.MAMBA: SimpleNamespace(value=canonical)}
    )
    component.tree_core = SimpleNamespace(node_by_id=lambda _: node)
    req = SimpleNamespace(kv=SimpleNamespace(mamba_cow_src_index=None))
    result = component.update_external_linker_load(
        ExternalLinkerLoadPhase.COMMIT,
        req,
        None,
        SimpleNamespace(device_indices=torch.tensor([7])),
        4,
        insert_result=SimpleNamespace(last_device_node=1, mamba_exist=True),
    )
    assert result is None
    assert req.kv.mamba_cow_src_index is canonical


@pytest.mark.parametrize("duplicate_checkpoint", [False, True])
@pytest.mark.parametrize("adopt_full_pages", [False, True])
def test_wrapper_commits_mamba_slot_without_token_page_filtering(
    duplicate_checkpoint, adopt_full_pages
):
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace(page_size=64)
    full = PoolTransfer(
        name=PoolName.KV, keys=["a", "b"], device_indices=torch.arange(128)
    )
    full_component = SimpleNamespace(
        component_type=ComponentType.FULL,
        update_external_linker_load=Mock(return_value=full),
    )
    checkpoint = PoolTransfer(
        name=PoolName.MAMBA, keys=["b"], device_indices=torch.tensor([7])
    )
    canonical = torch.tensor([9]) if duplicate_checkpoint else checkpoint.device_indices
    node = SimpleNamespace(
        component_data={ComponentType.MAMBA: SimpleNamespace(value=canonical)}
    )
    mamba = MambaComponent.__new__(MambaComponent)
    mamba.tree_core = SimpleNamespace(node_by_id=lambda _: node)
    req = SimpleNamespace(kv=SimpleNamespace(mamba_cow_src_index=None))
    result = wrapper._update_load(
        ExternalLinkerLoadPhase.COMMIT,
        req,
        [(full_component, full), (mamba, checkpoint)],
        prefix_len=128,
        insert_result=SimpleNamespace(
            adopted_ranges={
                ComponentType.FULL: [(0, 128)] if adopt_full_pages else [],
            },
            mamba_exist=duplicate_checkpoint,
            last_device_node=1,
        ),
        canonical_full=torch.arange(128),
    )
    expected = ([full] if adopt_full_pages else []) + (
        [] if duplicate_checkpoint else [checkpoint]
    )
    assert result == expected
    assert checkpoint.device_indices.tolist() == [7]
    assert checkpoint.keys == ["b"]
    assert req.kv.mamba_cow_src_index is canonical


class ControlledAdapter:
    def __init__(self, fail_submission=None):
        self.kvcr = self
        self.operations = []
        self.callbacks = {}
        self.fail_submission = fail_submission

    def deliver(self, blocks, request_id):
        if len(self.operations) == self.fail_submission:
            raise RuntimeError("injected submission failure")
        self.operations.append((blocks, request_id))
        return len(self.operations)

    def track(self, handle, callback):
        self.callbacks[handle] = callback

    def complete(self, handle, success=True):
        blocks, _ = self.operations[handle - 1]
        self.callbacks.pop(handle)(
            {key: SimpleNamespace(success=success) for key in blocks}
        )


def _remote_batch(*, requests=1, pages=2, **config):
    linker = KVCRDirectLinker.__new__(KVCRDirectLinker)
    linker.config = KVCRLinkerConfig(local_dram_bytes_per_worker=4096, **config)
    linker.num_layers = 3
    linker._lock = threading.RLock()
    linker.stats = collections.defaultdict(float)
    linker._adapter = ControlledAdapter()
    linker._restore_plans = {
        "kv": SimpleNamespace(
            order=(0, 1, 2),
            layer_slices=[(layer, layer, layer + 1) for layer in range(3)],
        )
    }
    linker._rows_for_pools = lambda pools: [indices.tolist() for _, indices in pools]
    linker._direct_remote_page_descriptors = lambda pool, row: tuple(range(3))
    linker._key = lambda page, pool: (pool, page)
    linker.layer_done_counter = LayerWiseLoadCounter(3)
    counter = linker.layer_done_counter.update_producer()
    pools = [
        _LoadPool("kv", list(range(pages)), torch.arange(pages), [], [], f"req{i}")
        for i in range(requests)
    ]
    batch = _LoadBatch(counter, [f"req{i}" for i in range(requests)], pools, None)
    finished = []
    linker._finish_load = lambda batch, error=None: finished.append(
        (batch.outstanding, batch.success, error)
    )
    return linker, batch, finished


def test_progressive_layer_waits_for_every_request_and_chunk():
    linker, batch, finished = _remote_batch(requests=2, direct_remote_chunk_pages=1)
    linker._submit_direct_remote_load(batch)
    adapter = linker._adapter
    assert len(adapter.operations) == 12
    futures = linker.layer_done_counter.futures[batch.counter_index]
    for handle in (1, 2, 3):
        adapter.complete(handle)
        assert not futures[0].done()
    adapter.complete(4)
    assert futures[0].done()
    assert not futures[1].done()
    for handle in range(5, 13):
        adapter.complete(handle)
    assert finished == [(0, True, None)]


@pytest.mark.parametrize("failure", ["entry", "submit", "refill"])
def test_remote_failure_drains_submitted_deliveries(failure):
    config = {"direct_remote_inflight_layers": 1} if failure == "refill" else {}
    linker, batch, finished = _remote_batch(**config)
    if failure in ("submit", "refill"):
        linker._adapter.fail_submission = 1
    linker._submit_direct_remote_load(batch)
    assert not finished
    adapter = linker._adapter
    adapter.complete(1, success=failure != "entry")
    for handle in list(adapter.callbacks):
        assert not finished
        adapter.complete(handle)
    assert finished[0][:2] == (0, False)
    assert len(finished) == 1


def test_remote_window_refills_without_releasing_layer_early():
    linker, batch, finished = _remote_batch(
        direct_remote_chunk_pages=1, direct_remote_inflight_layers=1
    )
    linker._submit_direct_remote_load(batch)
    adapter = linker._adapter
    assert len(adapter.operations) == 1
    adapter.complete(1)
    assert len(adapter.operations) == 2
    assert not linker.layer_done_counter.futures[batch.counter_index][0].done()
    for handle in range(2, 7):
        adapter.complete(handle)
    assert finished == [(0, True, None)]
