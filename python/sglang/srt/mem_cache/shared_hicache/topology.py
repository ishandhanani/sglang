from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional

from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_PLAN_VERSION,
    SharedHiCachePlan,
)
from sglang.srt.mem_cache.shared_hicache.transfer import (
    shared_hicache_parallel_rejection,
)


@dataclass(frozen=True)
class SharedHiCacheTopology:
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    attn_cp_rank: int = 0
    attn_cp_size: int = 1
    attn_tp_rank: int = 0
    attn_tp_size: int = 1
    attn_dp_rank: int = 0
    attn_dp_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1

    @classmethod
    def from_mapping(
        cls, parallel_metadata: Optional[Mapping[str, int]]
    ) -> "SharedHiCacheTopology":
        metadata = {key: int(value) for key, value in (parallel_metadata or {}).items()}
        tp_rank = int(metadata.get("tp_rank", 0))
        tp_size = int(metadata.get("tp_size", 1))
        return cls(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=int(metadata.get("pp_rank", 0)),
            pp_size=int(metadata.get("pp_size", 1)),
            attn_cp_rank=int(metadata.get("attn_cp_rank", 0)),
            attn_cp_size=int(metadata.get("attn_cp_size", 1)),
            attn_tp_rank=int(metadata.get("attn_tp_rank", tp_rank)),
            attn_tp_size=int(metadata.get("attn_tp_size", tp_size)),
            attn_dp_rank=int(metadata.get("attn_dp_rank", 0)),
            attn_dp_size=int(metadata.get("attn_dp_size", 1)),
            dp_rank=int(metadata.get("dp_rank", 0)),
            dp_size=int(metadata.get("dp_size", 1)),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def validate_target_rank(self, plan: SharedHiCachePlan) -> Optional[str]:
        topology_rejection = shared_hicache_parallel_rejection(
            pp_size=self.pp_size,
            attn_cp_size=self.attn_cp_size,
            attn_dp_size=self.attn_dp_size,
            tp_size=self.tp_size,
            attn_tp_size=self.attn_tp_size,
        )
        if topology_rejection is not None:
            return f"unsupported_target_topology:{topology_rejection}"

        if plan.target_tp_size != self.tp_size:
            return (
                f"wrong_target_tp_size:plan={plan.target_tp_size}:local={self.tp_size}"
            )
        if plan.source_tp_size != self.tp_size:
            return (
                "incompatible_source_tp_size:"
                f"source={plan.source_tp_size}:target={self.tp_size}"
            )
        target_tp_rank = (
            int(plan.target_tp_rank)
            if plan.target_tp_rank is not None
            else int(self.tp_rank)
        )
        if int(target_tp_rank) != self.tp_rank:
            return f"wrong_target_tp_rank:plan={target_tp_rank}:local={self.tp_rank}"
        source_tp_rank = plan.source_tp_rank
        if source_tp_rank is not None and int(source_tp_rank) != self.tp_rank:
            return f"wrong_source_tp_rank:plan={source_tp_rank}:local={self.tp_rank}"
        return None


def validate_shared_hicache_plan(
    plan: SharedHiCachePlan,
    *,
    worker_id: Optional[str],
    page_size: int,
    topology: SharedHiCacheTopology,
) -> Optional[str]:
    if worker_id is None:
        return "missing_worker_id"
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
    if not plan.is_shared_hicache():
        return "unsupported_source_medium"
    if plan.block_size_tokens != page_size:
        return "incompatible_block_size"
    return None
