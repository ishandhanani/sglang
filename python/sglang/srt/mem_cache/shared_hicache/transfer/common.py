from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import numpy as np

from sglang.srt.mem_cache.shared_hicache.config import (
    shared_hicache_transfer_backend_name,
)
from sglang.srt.mem_cache.shared_hicache.topology import SharedHiCacheTopology


class SharedHiCacheTransferBackend(ABC):
    name: str

    def __init__(
        self,
        *,
        target_session_id: str,
        target_kv_ptrs,
        target_kv_item_lens,
        target_num_pages: int,
        topology: SharedHiCacheTopology,
    ):
        if not self.name:
            raise ValueError("SharedHiCache transfer backend must define a name")
        self.target_session_id = str(target_session_id)
        self.target_kv_ptrs = [int(ptr) for ptr in target_kv_ptrs]
        self.target_kv_item_lens = [int(length) for length in target_kv_item_lens]
        self.target_num_pages = int(target_num_pages)
        self.topology = topology

    def target_descriptor(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "session_id": self.target_session_id,
            "target_num_pages": self.target_num_pages,
            **self.topology.to_dict(),
        }

    def local_gpu_id(self) -> Optional[int]:
        return None

    def create_source_worker(self):
        return self

    @abstractmethod
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
    ) -> None: ...

    def shutdown(self) -> None:
        pass


def make_shared_hicache_transfer_backend(
    scheduler,
    *,
    topology: SharedHiCacheTopology,
) -> SharedHiCacheTransferBackend:
    backend = shared_hicache_transfer_backend_name(scheduler.server_args)
    if backend != "nixl":
        raise RuntimeError(
            f"SharedHiCache transfer backend {backend!r} is not supported; "
            "specify --shared-hicache-transfer-backend nixl"
        )

    from sglang.srt.mem_cache.shared_hicache.transfer.nixl import (
        NixlSharedHiCacheTransferBackend,
    )
    topology_rejection = topology.unsupported_reason()
    if topology_rejection is not None:
        raise RuntimeError(topology_rejection)

    return NixlSharedHiCacheTransferBackend.from_scheduler(scheduler, topology=topology)
