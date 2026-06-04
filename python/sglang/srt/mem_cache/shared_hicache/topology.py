from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from sglang.srt.mem_cache.shared_hicache.plan import SharedHiCachePlan


@dataclass(frozen=True)
class SharedHiCacheTopology:
    """Local rank layout used to keep Shared HiCache transfers on matching TP ranks."""

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
    def from_scheduler(cls, scheduler) -> "SharedHiCacheTopology":
        ps = scheduler.ps
        dp_rank = 0 if ps.dp_rank is None else int(ps.dp_rank)
        return cls(
            tp_rank=int(ps.tp_rank),
            tp_size=int(ps.tp_size),
            pp_rank=int(ps.pp_rank),
            pp_size=int(ps.pp_size),
            attn_cp_rank=int(ps.attn_cp_rank),
            attn_cp_size=int(ps.attn_cp_size),
            attn_tp_rank=int(ps.attn_tp_rank),
            attn_tp_size=int(ps.attn_tp_size),
            attn_dp_rank=int(ps.attn_dp_rank),
            attn_dp_size=int(ps.attn_dp_size),
            dp_rank=dp_rank,
            dp_size=int(ps.dp_size),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def unsupported_reason(self) -> Optional[str]:
        unsupported = []
        if self.pp_size != 1:
            unsupported.append(f"pp_size={self.pp_size}")
        if self.attn_cp_size != 1:
            unsupported.append(f"attn_cp_size={self.attn_cp_size}")
        if self.attn_dp_size != 1:
            unsupported.append(f"attn_dp_size={self.attn_dp_size}")
        if self.tp_size != self.attn_tp_size:
            unsupported.append(
                f"tp_size={self.tp_size}:attn_tp_size={self.attn_tp_size}"
            )
        if not unsupported:
            return None
        return (
            "SharedHiCache direct transfer supports same-shape attention TP, but "
            "PP/CP/attention-DP "
            f"are deferred; got {', '.join(unsupported)}"
        )

    def validate_target_rank(self, plan: SharedHiCachePlan) -> Optional[str]:
        topology_rejection = self.unsupported_reason()
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

    def validate_source_rank(
        self,
        plan: SharedHiCachePlan,
        *,
        target_tp_rank: Optional[int],
        target_tp_size: Optional[int],
    ) -> Optional[str]:
        topology_rejection = self.unsupported_reason()
        if topology_rejection is not None:
            return f"unsupported_source_topology:{topology_rejection}"

        if plan.source_tp_size != self.tp_size:
            return (
                "wrong_source_tp_size:"
                f"plan={plan.source_tp_size}:local={self.tp_size}"
            )
        source_tp_rank = (
            int(plan.source_tp_rank)
            if plan.source_tp_rank is not None
            else int(self.tp_rank)
        )
        if source_tp_rank != self.tp_rank:
            return f"wrong_source_tp_rank:plan={source_tp_rank}:local={self.tp_rank}"
        if target_tp_size is None:
            return "missing_target_tp_size"
        if target_tp_rank is None:
            return "missing_target_tp_rank"

        target_tp_size = int(target_tp_size)
        target_tp_rank = int(target_tp_rank)
        if plan.target_tp_size != target_tp_size:
            return (
                "wrong_target_tp_size:"
                f"plan={plan.target_tp_size}:target={target_tp_size}"
            )
        plan_target_tp_rank = (
            int(plan.target_tp_rank)
            if plan.target_tp_rank is not None
            else target_tp_rank
        )
        if plan_target_tp_rank != target_tp_rank:
            return (
                "wrong_target_tp_rank:"
                f"plan={plan_target_tp_rank}:target={target_tp_rank}"
            )
        if target_tp_size != self.tp_size:
            return f"incompatible_tp_size:source={self.tp_size}:target={target_tp_size}"
        if target_tp_rank != self.tp_rank:
            return (
                "wrong_source_tp_rank_for_target:"
                f"source={self.tp_rank}:target={target_tp_rank}"
            )
        return None
