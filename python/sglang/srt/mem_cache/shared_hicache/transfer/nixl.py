from __future__ import annotations

import base64
import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np

from sglang.srt.environ import envs
from sglang.srt.mem_cache.shared_hicache.topology import SharedHiCacheTopology
from sglang.srt.mem_cache.shared_hicache.transfer.common import (
    SharedHiCacheTransferBackend,
)

logger = logging.getLogger(__name__)

_SHARED_HICACHE_NIXL_NOTIFICATION_PREFIX = "shared_hicache:"

if TYPE_CHECKING:
    import torch


def _build_completion_notification(
    *,
    transfer_id: str,
    transferred_blocks: int,
    reason: str,
) -> str:
    payload = {
        "transfer_id": str(transfer_id),
        "transferred_blocks": int(transferred_blocks),
        "reason": str(reason),
    }
    return _SHARED_HICACHE_NIXL_NOTIFICATION_PREFIX + json.dumps(
        payload, separators=(",", ":")
    )


def _parse_completion_notification(
    message: bytes | str,
) -> Optional[tuple[str, int, str]]:
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if not isinstance(message, str) or not message.startswith(
        _SHARED_HICACHE_NIXL_NOTIFICATION_PREFIX
    ):
        return None
    try:
        payload = json.loads(message[len(_SHARED_HICACHE_NIXL_NOTIFICATION_PREFIX) :])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    transfer_id = str(payload.get("transfer_id") or "")
    if not transfer_id:
        return None
    try:
        transferred_blocks = int(payload.get("transferred_blocks", 0))
    except (TypeError, ValueError):
        return None
    if transferred_blocks < 0:
        return None
    return transfer_id, transferred_blocks, str(payload.get("reason", "ok"))


def _target_kv_pool_from_scheduler(scheduler):
    target_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if hasattr(target_pool, "full_kv_pool") and hasattr(target_pool, "full_layer_nums"):
        raise RuntimeError(
            "SharedHiCache direct transfer does not support hybrid linear-attention KV pools"
        )
    return target_pool


def _scheduler_gpu_id(scheduler) -> int:
    return int(scheduler.ps.gpu_id)


def _nixl_backend_params(backend: str, transfer_parallelism: int) -> dict[str, str]:
    backend_params = json.loads(envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS.get())
    if not isinstance(backend_params, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in backend_params.items()
    ):
        raise ValueError(
            "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS must be a JSON object "
            "with string keys and string values"
        )
    if transfer_parallelism > 0:
        if backend in {"UCX", "OBJ"}:
            backend_params.setdefault("num_threads", str(transfer_parallelism))
        elif backend == "GDS_MT":
            backend_params.setdefault("thread_count", str(transfer_parallelism))
        elif backend == "UCCL":
            backend_params.setdefault("num_cpus", str(transfer_parallelism))
    return backend_params


def _create_nixl_agent(*, transfer_parallelism: int):
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as err:
        raise ImportError(
            "Please install NIXL by following the instructions at "
            "https://github.com/ai-dynamo/nixl/blob/main/README.md "
            "to use SharedHiCache NIXL direct transfer."
        ) from err

    backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
    backend_params = _nixl_backend_params(backend, transfer_parallelism)
    agent_config = nixl_agent_config(
        backends=[],
        num_threads=max(0, int(transfer_parallelism)),
    )
    agent_name = f"shared_hicache_nixl_{uuid.uuid4()}"
    agent = nixl_agent(agent_name, agent_config)
    agent.create_backend(backend, backend_params)
    available_plugins = agent.get_plugin_list()
    if backend not in available_plugins:
        raise RuntimeError(
            f"NIXL backend {backend!r} not found. Available: {available_plugins}"
        )
    return agent, agent_name, backend


def _metadata_log_value(metadata: Optional[Mapping[str, Any]], field_name: str) -> Any:
    if not isinstance(metadata, Mapping):
        return "n/a"
    value = metadata.get(field_name)
    return "n/a" if value is None else value


def _num_pages_from_buf_infos(
    data_lens: list[int], item_lens: list[int], *, label: str
) -> int:
    if not data_lens or len(data_lens) != len(item_lens):
        raise RuntimeError(f"{label} KV buffer metadata is malformed")
    pages = []
    for data_len, item_len in zip(data_lens, item_lens):
        if item_len <= 0 or data_len <= 0 or data_len % item_len != 0:
            raise RuntimeError(
                f"{label} KV buffer length is not page-aligned: "
                f"data_len={data_len} item_len={item_len}"
            )
        pages.append(int(data_len // item_len))
    first = pages[0]
    if any(page != first for page in pages):
        raise RuntimeError(f"{label} KV buffers have mismatched page counts")
    return first


def _source_host_buf_infos(tree_cache) -> tuple[list[int], list[int], list[int], int]:
    refs = _source_host_tensors(tree_cache)
    page_size = tree_cache.page_size
    ptrs = [int(ref.data_ptr()) for ref in refs]
    data_lens = [int(ref.nbytes) for ref in refs]
    item_lens = [int(ref[0].nbytes) * page_size for ref in refs]
    num_pages = _num_pages_from_buf_infos(data_lens, item_lens, label="source host")
    return ptrs, data_lens, item_lens, num_pages


def _source_host_tensors(tree_cache) -> list[torch.Tensor]:
    host_pool = tree_cache.cache_controller.mem_pool_host
    if getattr(host_pool, "layout", None) != "layer_first":
        raise RuntimeError(
            "SharedHiCache direct transfer requires layer_first host layout, "
            f"got {getattr(host_pool, 'layout', None)!r}"
        )

    if hasattr(host_pool, "k_data_refs") and hasattr(host_pool, "v_data_refs"):
        refs = host_pool.k_data_refs + host_pool.v_data_refs
    elif hasattr(host_pool, "data_refs"):
        refs = host_pool.data_refs
    else:
        raise RuntimeError(
            "Unsupported HiCache host pool for SharedHiCache direct transfer"
        )
    return list(refs)


def _validate_kv_item_lens_match(
    source_kv_item_lens: list[int], target_kv_item_lens: list[int]
) -> None:
    if len(source_kv_item_lens) != len(target_kv_item_lens):
        raise RuntimeError(
            "KV item length count mismatch: "
            f"source={len(source_kv_item_lens)} target={len(target_kv_item_lens)}"
        )

    src_item_lens = np.asarray(source_kv_item_lens, dtype=np.uint64)
    dst_item_lens = np.asarray(target_kv_item_lens, dtype=np.uint64)
    mismatched_items = np.nonzero(src_item_lens != dst_item_lens)[0]
    if len(mismatched_items) > 0:
        idx = int(mismatched_items[0])
        raise RuntimeError(
            "KV item length mismatch: "
            f"source={int(src_item_lens[idx])} target={int(dst_item_lens[idx])}"
        )


@dataclass
class _NixlPreppedTarget:
    handle: Any
    num_pages: int


@dataclass
class _NixlSourceWorkerState:
    agent: Any
    agent_name: str
    backend_name: str
    source_prep_handle: Any
    source_num_pages: int
    source_kv_item_lens: list[int]
    remote_agents: set[str] = field(default_factory=set)
    target_prep_handles: dict[str, _NixlPreppedTarget] = field(default_factory=dict)


class NixlSharedHiCacheTransferBackend(SharedHiCacheTransferBackend):
    """NIXL-backed source-HiCache-host to target-GPU-device transfer helper."""

    name = "nixl"

    def __init__(
        self,
        *,
        agent,
        agent_name: str,
        backend_name: str,
        tree_cache,
        target_kv_ptrs,
        target_kv_item_lens,
        target_num_pages: int,
        gpu_id: int,
        topology: SharedHiCacheTopology,
        transfer_parallelism: int,
    ):
        super().__init__(
            target_session_id=agent_name,
            target_kv_ptrs=target_kv_ptrs,
            target_kv_item_lens=target_kv_item_lens,
            target_num_pages=target_num_pages,
            topology=topology,
        )
        self.agent = agent
        self.agent_name = agent_name
        self.backend_name = backend_name
        self.tree_cache = tree_cache
        self._target_notification_lock = threading.Lock()
        self._target_notifications: dict[str, tuple[int, str]] = {}
        self._retired_target_notifications: set[str] = set()
        self._retired_target_notification_order: deque[str] = deque()
        self._gpu_id = int(gpu_id)
        self._transfer_parallelism = int(transfer_parallelism)
        self._shutdown = False

    @classmethod
    def from_scheduler(
        cls, scheduler, *, topology: SharedHiCacheTopology
    ) -> "NixlSharedHiCacheTransferBackend":
        transfer_parallelism = max(
            1, int(envs.SGLANG_SHARED_HICACHE_TRANSFER_PARALLELISM.get())
        )
        agent, agent_name, backend_name = _create_nixl_agent(
            transfer_parallelism=transfer_parallelism
        )
        target_pool = _target_kv_pool_from_scheduler(scheduler)
        target_kv_ptrs, target_kv_lens, target_kv_item_lens = (
            target_pool.get_contiguous_buf_infos()
        )
        target_num_pages = _num_pages_from_buf_infos(
            [int(length) for length in target_kv_lens],
            [int(length) for length in target_kv_item_lens],
            label="target device",
        )
        gpu_id = _scheduler_gpu_id(scheduler)
        target_descs = agent.register_memory(
            [
                (int(ptr), int(length), gpu_id, "")
                for ptr, length in zip(target_kv_ptrs, target_kv_lens)
            ],
            "VRAM",
        )
        if not target_descs:
            raise RuntimeError("SharedHiCache NIXL target KV registration failed")
        transfer = cls(
            agent=agent,
            agent_name=agent_name,
            backend_name=backend_name,
            tree_cache=scheduler.tree_cache,
            target_kv_ptrs=target_kv_ptrs,
            target_kv_item_lens=target_kv_item_lens,
            target_num_pages=target_num_pages,
            gpu_id=gpu_id,
            topology=topology,
            transfer_parallelism=transfer_parallelism,
        )
        transfer._validate_source_host_pool()
        transfer._log_ready()
        return transfer

    def target_descriptor(self) -> dict[str, Any]:
        descriptor = super().target_descriptor()
        descriptor.update(
            {
                "agent_name": self.agent_name,
                "agent_metadata": base64.b64encode(
                    self.agent.get_agent_metadata()
                ).decode("ascii"),
                "gpu_id": self._gpu_id,
                "transport": {
                    "backend": self.backend_name,
                    "transfer_parallelism": self._transfer_parallelism,
                },
            }
        )
        return descriptor

    def local_gpu_id(self) -> int:
        return self._gpu_id

    def _log_ready(self) -> None:
        logger.info(
            "SharedHiCache NIXL direct transfer enabled agent=%s backend=%s "
            "tp_rank=%d gpu_id=%d parallelism=%d",
            self.agent_name,
            self.backend_name,
            self.topology.tp_rank,
            self._gpu_id,
            self._transfer_parallelism,
        )

    def _validate_source_host_pool(self) -> None:
        host_pool = self.tree_cache.cache_controller.mem_pool_host
        if not hasattr(host_pool, "kv_buffer"):
            raise RuntimeError(
                "SharedHiCache NIXL direct transfer requires a host kv_buffer"
            )
        _, _, source_kv_item_lens, _ = _source_host_buf_infos(self.tree_cache)
        _validate_kv_item_lens_match(source_kv_item_lens, self.target_kv_item_lens)

    def _prep_dlist(
        self,
        agent,
        *,
        peer_name: str,
        ptrs: list[int],
        item_lens: list[int],
        num_pages: int,
        location: str,
        device_id: int,
    ):
        arrays = []
        for ptr, item_len in zip(ptrs, item_lens):
            addrs = np.arange(num_pages, dtype=np.int64) * int(item_len) + int(ptr)
            arrays.append(
                np.column_stack(
                    [
                        addrs,
                        np.full(num_pages, int(item_len), dtype=np.int64),
                        np.full(num_pages, int(device_id), dtype=np.int64),
                    ]
                )
            )
        handle = agent.prep_xfer_dlist(peer_name, np.vstack(arrays), location)
        if handle is None:
            raise RuntimeError("NIXL direct KV transfer descriptor preparation failed")
        return handle

    def _create_source_worker_state(self) -> _NixlSourceWorkerState:
        host_pool = self.tree_cache.cache_controller.mem_pool_host
        agent, agent_name, backend_name = _create_nixl_agent(
            transfer_parallelism=self._transfer_parallelism
        )
        source_descs = agent.register_memory(
            [
                (
                    int(host_pool.kv_buffer.data_ptr()),
                    int(host_pool.kv_buffer.nbytes),
                    0,
                    "",
                )
            ],
            "DRAM",
        )
        if not source_descs:
            raise RuntimeError("SharedHiCache NIXL source host registration failed")
        source_kv_ptrs, _, source_kv_item_lens, source_num_pages = (
            _source_host_buf_infos(self.tree_cache)
        )
        source_prep_handle = self._prep_dlist(
            agent,
            peer_name="",
            ptrs=source_kv_ptrs,
            item_lens=source_kv_item_lens,
            num_pages=source_num_pages,
            location="DRAM",
            device_id=0,
        )
        logger.info(
            "SharedHiCache NIXL source transfer worker enabled agent=%s backend=%s "
            "tp_rank=%d gpu_id=%d thread=%d parallelism=%d source_pages=%d",
            agent_name,
            backend_name,
            self.topology.tp_rank,
            self._gpu_id,
            threading.get_ident(),
            self._transfer_parallelism,
            source_num_pages,
        )
        return _NixlSourceWorkerState(
            agent=agent,
            agent_name=agent_name,
            backend_name=backend_name,
            source_prep_handle=source_prep_handle,
            source_num_pages=source_num_pages,
            source_kv_item_lens=source_kv_item_lens,
        )

    def create_source_worker(self):
        if self._shutdown:
            raise RuntimeError("NIXL direct KV transfer backend is not enabled")
        return _NixlSourceTransferWorker(self, self._create_source_worker_state())

    def _add_remote_target(
        self,
        state: _NixlSourceWorkerState,
        target_metadata: Optional[Mapping[str, Any]],
        *,
        target_kv_ptrs: list[int],
        target_kv_item_lens: list[int],
        target_num_pages: int,
    ) -> str:
        if not isinstance(target_metadata, Mapping):
            raise RuntimeError("NIXL target metadata must be an object")
        target_agent_name = str(target_metadata.get("agent_name") or "")
        encoded_metadata = target_metadata.get("agent_metadata")
        if not target_agent_name:
            raise RuntimeError("NIXL target metadata missing agent_name")
        if not isinstance(encoded_metadata, str) or not encoded_metadata:
            raise RuntimeError("NIXL target metadata missing agent_metadata")
        if target_agent_name not in state.remote_agents:
            state.agent.add_remote_agent(base64.b64decode(encoded_metadata))
            state.remote_agents.add(target_agent_name)
        if target_agent_name not in state.target_prep_handles:
            target_gpu_id_raw = target_metadata.get("gpu_id")
            if isinstance(target_gpu_id_raw, bool) or not isinstance(
                target_gpu_id_raw, (int, np.integer)
            ):
                raise RuntimeError("NIXL target metadata missing gpu_id")
            state.target_prep_handles[target_agent_name] = _NixlPreppedTarget(
                handle=self._prep_dlist(
                    state.agent,
                    peer_name=target_agent_name,
                    ptrs=target_kv_ptrs,
                    item_lens=target_kv_item_lens,
                    num_pages=target_num_pages,
                    location="VRAM",
                    device_id=int(target_gpu_id_raw),
                ),
                num_pages=target_num_pages,
            )
        return target_agent_name

    def _wait_for_transfer(self, agent, handle) -> None:
        try:
            transfer_state = agent.transfer(handle)
            while True:
                if transfer_state == "ERR":
                    raise RuntimeError("NIXL direct KV transfer failed")
                if transfer_state == "DONE":
                    break
                time.sleep(0)
                transfer_state = agent.check_xfer_state(handle)
        finally:
            try:
                agent.release_xfer_handle(handle)
            except Exception:
                logger.debug(
                    "SharedHiCache NIXL transfer handle release failed",
                    exc_info=True,
                )

    def _drain_target_notifications_locked(self) -> None:
        try:
            notif_map = self.agent.get_new_notifs()
        except Exception:
            logger.debug(
                "SharedHiCache NIXL target notification poll failed", exc_info=True
            )
            return
        if not isinstance(notif_map, Mapping):
            return
        for messages in notif_map.values():
            for message in messages or ():
                parsed = _parse_completion_notification(message)
                if parsed is None:
                    continue
                transfer_id, transferred_blocks, reason = parsed
                if transfer_id in self._retired_target_notifications:
                    continue
                self._target_notifications[transfer_id] = (
                    transferred_blocks,
                    reason,
                )

    def pop_target_transfer_notification(
        self, transfer_id: str
    ) -> Optional[tuple[int, str]]:
        with self._target_notification_lock:
            self._drain_target_notifications_locked()
            return self._target_notifications.pop(str(transfer_id), None)

    def drop_target_transfer_notification(self, transfer_id: str) -> None:
        transfer_id = str(transfer_id)
        with self._target_notification_lock:
            self._target_notifications.pop(transfer_id, None)
            if transfer_id not in self._retired_target_notifications:
                self._retired_target_notifications.add(transfer_id)
                self._retired_target_notification_order.append(transfer_id)
            while len(self._retired_target_notification_order) > 4096:
                expired = self._retired_target_notification_order.popleft()
                self._retired_target_notifications.discard(expired)

    def transfer_pages(
        self,
        *,
        transfer_id: str,
        plan_id: str,
        transferred_blocks: int,
        completion_reason: str,
        source_page_indices: np.ndarray,
        target_page_indices: np.ndarray,
        target_kv_ptrs: list[int],
        target_kv_item_lens: list[int],
        target_num_pages: int,
        target_metadata: Optional[Mapping[str, Any]] = None,
        x_request_id: Optional[str] = None,
    ) -> None:
        if self._shutdown:
            raise RuntimeError("NIXL direct KV transfer backend is not enabled")
        self._transfer_pages_from_state(
            self._create_source_worker_state(),
            transfer_id=transfer_id,
            plan_id=plan_id,
            transferred_blocks=transferred_blocks,
            completion_reason=completion_reason,
            source_page_indices=source_page_indices,
            target_page_indices=target_page_indices,
            target_kv_ptrs=target_kv_ptrs,
            target_kv_item_lens=target_kv_item_lens,
            target_num_pages=target_num_pages,
            target_metadata=target_metadata,
            x_request_id=x_request_id,
        )

    def _transfer_pages_from_state(
        self,
        source_state: _NixlSourceWorkerState,
        *,
        transfer_id: str,
        plan_id: str,
        transferred_blocks: int,
        completion_reason: str,
        source_page_indices: np.ndarray,
        target_page_indices: np.ndarray,
        target_kv_ptrs: list[int],
        target_kv_item_lens: list[int],
        target_num_pages: int,
        target_metadata: Optional[Mapping[str, Any]] = None,
        x_request_id: Optional[str] = None,
    ) -> None:
        if self._shutdown:
            raise RuntimeError("NIXL direct KV transfer backend is not enabled")
        setup_start = time.perf_counter()
        target_agent_name = self._add_remote_target(
            source_state,
            target_metadata,
            target_kv_ptrs=target_kv_ptrs,
            target_kv_item_lens=target_kv_item_lens,
            target_num_pages=target_num_pages,
        )
        if source_page_indices.size == 0:
            return
        if int(source_page_indices.max()) >= source_state.source_num_pages:
            raise RuntimeError("NIXL source page index out of prepared range")
        if int(target_page_indices.max()) >= target_num_pages:
            raise RuntimeError("NIXL target page index out of prepared range")
        _validate_kv_item_lens_match(
            source_state.source_kv_item_lens, target_kv_item_lens
        )
        if len(source_state.source_kv_item_lens) != len(target_kv_ptrs):
            raise RuntimeError(
                "KV pointer count mismatch: "
                f"source={len(source_state.source_kv_item_lens)} "
                f"target={len(target_kv_ptrs)}"
            )
        target_prep = source_state.target_prep_handles[target_agent_name]
        if target_prep.num_pages != target_num_pages:
            raise RuntimeError("NIXL target page count changed after preparation")
        num_layers = len(source_state.source_kv_item_lens)
        layer_offsets = np.arange(num_layers, dtype=np.int32)
        source_indices = (
            layer_offsets[:, None] * source_state.source_num_pages
            + source_page_indices.astype(np.int32, copy=False)[None, :]
        ).ravel()
        target_indices = (
            layer_offsets[:, None] * target_num_pages
            + target_page_indices.astype(np.int32, copy=False)[None, :]
        ).ravel()
        completion_notification = _build_completion_notification(
            transfer_id=transfer_id,
            transferred_blocks=transferred_blocks,
            reason=completion_reason,
        )
        handle = source_state.agent.make_prepped_xfer(
            "WRITE",
            source_state.source_prep_handle,
            source_indices,
            target_prep.handle,
            target_indices,
            completion_notification.encode("utf-8"),
        )
        if not handle:
            raise RuntimeError("NIXL direct KV transfer initialization failed")
        setup_ms = (time.perf_counter() - setup_start) * 1000
        start = time.perf_counter()
        self._wait_for_transfer(source_state.agent, handle)
        transfer_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "SharedHiCache NIXL transferred transfer_id=%s plan_id=%s "
            "x_request_id=%s blocks=%d slices=%d bytes=%d ms=%.3f "
            "setup_ms=%.3f source_agent=%s source_tp_rank=%d source_gpu_id=%d "
            "target_tp_rank=%s target_gpu_id=%s prepped=true",
            transfer_id,
            plan_id,
            x_request_id,
            len(source_page_indices),
            int(source_indices.size),
            int(len(source_page_indices) * sum(source_state.source_kv_item_lens)),
            transfer_ms,
            setup_ms,
            source_state.agent_name,
            self.topology.tp_rank,
            self._gpu_id,
            _metadata_log_value(target_metadata, "tp_rank"),
            _metadata_log_value(target_metadata, "gpu_id"),
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._target_notifications = {}
        self._retired_target_notifications = set()
        self._retired_target_notification_order = deque()


class _NixlSourceTransferWorker:
    def __init__(
        self,
        owner: NixlSharedHiCacheTransferBackend,
        state: _NixlSourceWorkerState,
    ):
        self._owner = owner
        self._state = state

    def transfer_pages(
        self,
        *,
        transfer_id: str,
        plan_id: str,
        transferred_blocks: int,
        completion_reason: str,
        source_page_indices: np.ndarray,
        target_page_indices: np.ndarray,
        target_kv_ptrs: list[int],
        target_kv_item_lens: list[int],
        target_num_pages: int,
        target_metadata: Optional[Mapping[str, Any]] = None,
        x_request_id: Optional[str] = None,
    ) -> None:
        self._owner._transfer_pages_from_state(
            self._state,
            transfer_id=transfer_id,
            plan_id=plan_id,
            transferred_blocks=transferred_blocks,
            completion_reason=completion_reason,
            source_page_indices=source_page_indices,
            target_page_indices=target_page_indices,
            target_kv_ptrs=target_kv_ptrs,
            target_kv_item_lens=target_kv_item_lens,
            target_num_pages=target_num_pages,
            target_metadata=target_metadata,
            x_request_id=x_request_id,
        )
