from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import numpy as np

from sglang.srt.mem_cache.shared_hicache.config import (
    shared_hicache_transfer_backend_name,
)


class SharedHiCacheTransferBackend(ABC):
    name: str

    def __init__(
        self,
        *,
        target_session_id: str,
        target_kv_ptrs,
        target_kv_item_lens,
        parallel_metadata: Optional[Mapping[str, int]] = None,
    ):
        if not getattr(self, "name", None):
            raise ValueError("SharedHiCache transfer backend must define a name")
        self.target_session_id = str(target_session_id)
        self.target_kv_ptrs = [int(ptr) for ptr in target_kv_ptrs]
        self.target_kv_item_lens = [int(length) for length in target_kv_item_lens]
        self.parallel_metadata = {
            key: int(value) for key, value in (parallel_metadata or {}).items()
        }

    def target_descriptor(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "session_id": self.target_session_id,
            **self.parallel_metadata,
        }

    def prepare_source_worker(self) -> None:
        pass

    @abstractmethod
    def transfer_pages(
        self,
        *,
        transfer_id: str,
        transferred_blocks: int,
        completion_reason: str,
        source_page_indices: np.ndarray,
        target_page_indices: np.ndarray,
        target_kv_ptrs: list[int],
        target_kv_item_lens: list[int],
        target_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None: ...

    def shutdown(self) -> None:
        pass


def make_shared_hicache_transfer_backend(
    scheduler,
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
    from sglang.srt.mem_cache.shared_hicache.topology import (
        shared_hicache_topology_rejection_from_scheduler,
    )

    topology_rejection = shared_hicache_topology_rejection_from_scheduler(scheduler)
    if topology_rejection is not None:
        raise RuntimeError(topology_rejection)

    return NixlSharedHiCacheTransferBackend.from_scheduler(scheduler)
