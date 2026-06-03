from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Optional

from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_PLAN_VERSION,
    SharedHiCachePlan,
)


def _server_arg(scheduler, name: str, default: int) -> int:
    server_args = getattr(scheduler, "server_args", None)
    value = getattr(server_args, name, default)
    return int(value if value is not None else default)


def _parallel_value(scheduler, name: str, default: int) -> int:
    ps = getattr(scheduler, "ps", None)
    if ps is not None and hasattr(ps, name):
        value = getattr(ps, name)
    else:
        value = getattr(scheduler, name, default)
    return int(value if value is not None else default)


def scheduler_parallel_metadata(scheduler) -> dict[str, int]:
    """Return rank metadata needed for same-shape direct reuse."""

    return {
        "tp_rank": _parallel_value(scheduler, "tp_rank", 0),
        "tp_size": _parallel_value(
            scheduler, "tp_size", _server_arg(scheduler, "tp_size", 1)
        ),
        "pp_rank": _parallel_value(scheduler, "pp_rank", 0),
        "pp_size": _parallel_value(
            scheduler, "pp_size", _server_arg(scheduler, "pp_size", 1)
        ),
        "attn_cp_rank": _parallel_value(scheduler, "attn_cp_rank", 0),
        "attn_cp_size": _parallel_value(
            scheduler, "attn_cp_size", _server_arg(scheduler, "attn_cp_size", 1)
        ),
        "attn_tp_rank": _parallel_value(scheduler, "attn_tp_rank", 0),
        "attn_tp_size": _parallel_value(
            scheduler, "attn_tp_size", _server_arg(scheduler, "tp_size", 1)
        ),
        "attn_dp_rank": _parallel_value(scheduler, "attn_dp_rank", 0),
        "attn_dp_size": _parallel_value(
            scheduler, "attn_dp_size", _server_arg(scheduler, "dp_size", 1)
        ),
        "dp_rank": _parallel_value(scheduler, "dp_rank", 0),
        "dp_size": _parallel_value(
            scheduler, "dp_size", _server_arg(scheduler, "dp_size", 1)
        ),
    }


def shared_hicache_parallel_rejection(
    *,
    pp_size: int,
    attn_cp_size: int,
    attn_dp_size: int = 1,
    tp_size: Optional[int] = None,
    attn_tp_size: Optional[int] = None,
) -> Optional[str]:
    unsupported = []
    if pp_size != 1:
        unsupported.append(f"pp_size={pp_size}")
    if attn_cp_size != 1:
        unsupported.append(f"attn_cp_size={attn_cp_size}")
    if attn_dp_size != 1:
        unsupported.append(f"attn_dp_size={attn_dp_size}")
    if (
        tp_size is not None
        and attn_tp_size is not None
        and int(tp_size) != int(attn_tp_size)
    ):
        unsupported.append(f"tp_size={tp_size}:attn_tp_size={attn_tp_size}")
    if unsupported:
        return (
            "SharedHiCache direct transfer supports same-shape attention TP, but "
            "PP/CP/attention-DP "
            f"are deferred; got {', '.join(unsupported)}"
        )
    return None


def shared_hicache_topology_rejection_from_scheduler(scheduler) -> Optional[str]:
    return shared_hicache_parallel_rejection(
        pp_size=_server_arg(scheduler, "pp_size", 1),
        attn_cp_size=_server_arg(scheduler, "attn_cp_size", 1),
        attn_dp_size=_parallel_value(
            scheduler, "attn_dp_size", _server_arg(scheduler, "dp_size", 1)
        ),
        tp_size=_server_arg(scheduler, "tp_size", 1),
        attn_tp_size=_parallel_value(
            scheduler, "attn_tp_size", _server_arg(scheduler, "tp_size", 1)
        ),
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
