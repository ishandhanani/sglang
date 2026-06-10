from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.mem_cache.shared_hicache.plan import SharedHiCachePlan

if TYPE_CHECKING:
    from sglang.srt.mem_cache.shared_hicache.control import SharedHiCacheTransferHandle


@dataclass
class SharedHiCachePendingFetch:
    plan: SharedHiCachePlan
    plan_offset: int
    target_start_block: int
    expected_hashes: tuple[int, ...]
    transfer: SharedHiCacheTransferHandle
    device_indices: Optional[torch.Tensor] = None
    locked_node: Optional[TreeNode] = None
    backend: str = "unknown"
    bytes_per_page: int = 0
    submitted_at: float = 0.0
    done_at: float = 0.0
