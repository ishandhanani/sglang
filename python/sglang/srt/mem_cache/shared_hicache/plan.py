from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

from sglang.srt.disaggregation.kv_events import StorageMedium

SHARED_HICACHE_PLAN_VERSION = 1
SHARED_HICACHE_DIRECT_TIMEOUT_REASON = "source_transfer_timeout_maybe_inflight"
SHARED_HICACHE_SOURCE_MEDIUM = StorageMedium.CPU.value
_SIGNED_INT64_MIN = -(2**63)
_SIGNED_INT64_MAX = 2**63 - 1
_UINT64_MAX = 2**64 - 1


def _now_ms() -> int:
    return int(time.time() * 1000)


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        return endpoint
    if "://" not in endpoint:
        endpoint = f"tcp://{endpoint}"
    if not endpoint.startswith("tcp://"):
        raise ValueError("shared HiCache control endpoint must use tcp://")
    return endpoint.rstrip("/")


def _canonical_source_medium(medium: Any) -> str:
    if medium != SHARED_HICACHE_SOURCE_MEDIUM:
        raise ValueError(
            f"source_medium must be {SHARED_HICACHE_SOURCE_MEDIUM!r}, got {medium!r}"
        )
    return SHARED_HICACHE_SOURCE_MEDIUM


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return int(value)
    raise ValueError(f"{field_name} must be an integer, got {value!r}")


def _coerce_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _coerce_optional_int(value: Any, field_name: str) -> Optional[int]:
    if value is None:
        return None
    return _coerce_int(value, field_name)


def _coerce_positive_int(value: Any, field_name: str) -> int:
    value = _coerce_int(value, field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _coerce_port(value: Any, field_name: str) -> int:
    port = _coerce_int(value, field_name)
    if port <= 0 or port > 65535:
        raise ValueError(f"{field_name} must be in [1, 65535]")
    return port


def _coerce_array(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{field_name} must be an array")
    try:
        return list(value)
    except TypeError as err:
        raise ValueError(f"{field_name} must be an array") from err


def _coerce_block_hash(value: Any, field_name: str = "block_hash") -> int:
    value = _coerce_int(value, field_name)
    if _SIGNED_INT64_MIN <= value <= _SIGNED_INT64_MAX:
        return value
    if 0 <= value <= _UINT64_MAX:
        # Shared HiCache plans cross a JSON boundary from routers that use u64
        # internally, while SGLang KV events and host indexing use signed int64.
        # Normalize once here so source-side cache lookup remains exact.
        return value - 2**64
    raise ValueError(f"{field_name} must fit in signed int64 or uint64")


@dataclass(frozen=True)
class SharedHiCachePlan:
    plan_id: str
    request_id: str
    target_worker_id: str
    source_worker_id: str
    source_host: str
    source_bootstrap_port: int
    source_medium: str
    router_block_hashes: tuple[int, ...]
    engine_block_hashes: tuple[int, ...]
    planned_prefix_blocks: int
    block_size_tokens: int
    created_at_ms: int
    expires_at_ms: int
    start_block_index: int = 0
    plan_version: int = SHARED_HICACHE_PLAN_VERSION
    source_tp_rank: Optional[int] = None
    source_tp_size: int = 1
    target_tp_rank: Optional[int] = None
    target_tp_size: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SharedHiCachePlan":
        if not isinstance(data, Mapping):
            raise ValueError("SharedHiCache plan must be a mapping")

        if "router_block_hashes" not in data:
            raise ValueError("SharedHiCache plan missing router_block_hashes")
        router_block_hashes = tuple(
            _coerce_block_hash(item, "router_block_hash")
            for item in _coerce_array(
                data["router_block_hashes"], "router_block_hashes"
            )
        )
        if "engine_block_hashes" not in data:
            raise ValueError("SharedHiCache plan missing engine_block_hashes")
        # These arrays are parallel but intentionally separate today:
        # - router_block_hashes are router/request block identities. They
        #   preserve plan order and label the pages returned to the target.
        # - engine_block_hashes are source-worker HostPinned lookup keys from
        #   framework KV events. The source host index is keyed by these values.
        # If Dynamo and SGLang later share one canonical block-hash contract
        # (same representation, algorithm, and parent-chaining semantics), this
        # source-lookup field can collapse into router_block_hashes.
        engine_block_hashes = tuple(
            _coerce_block_hash(item, "engine_block_hash")
            for item in _coerce_array(
                data["engine_block_hashes"], "engine_block_hashes"
            )
        )
        if len(engine_block_hashes) != len(router_block_hashes):
            raise ValueError(
                "engine_block_hashes length must match router_block_hashes"
            )

        planned_prefix_blocks = _coerce_int(
            data.get("planned_prefix_blocks", len(router_block_hashes)),
            "planned_prefix_blocks",
        )
        if planned_prefix_blocks < 0:
            raise ValueError("planned_prefix_blocks must be non-negative")
        start_block_index = _coerce_int(
            data.get("start_block_index", 0),
            "start_block_index",
        )
        if start_block_index < 0:
            raise ValueError("start_block_index must be non-negative")

        try:
            plan = cls(
                plan_id=str(data.get("plan_id", "")),
                request_id=str(data.get("request_id", "")),
                target_worker_id=_coerce_string(
                    data["target_worker_id"], "target_worker_id"
                ),
                source_worker_id=_coerce_string(
                    data["source_worker_id"], "source_worker_id"
                ),
                source_host=_coerce_string(data["source_host"], "source_host"),
                source_bootstrap_port=_coerce_port(
                    data["source_bootstrap_port"],
                    "source_bootstrap_port",
                ),
                source_medium=_canonical_source_medium(data["source_medium"]),
                router_block_hashes=router_block_hashes,
                engine_block_hashes=engine_block_hashes,
                planned_prefix_blocks=min(
                    planned_prefix_blocks, len(router_block_hashes)
                ),
                block_size_tokens=_coerce_int(
                    data["block_size_tokens"],
                    "block_size_tokens",
                ),
                created_at_ms=_coerce_int(
                    data.get("created_at_ms", 0), "created_at_ms"
                ),
                expires_at_ms=_coerce_int(data["expires_at_ms"], "expires_at_ms"),
                start_block_index=start_block_index,
                plan_version=_coerce_int(
                    data.get("plan_version", SHARED_HICACHE_PLAN_VERSION),
                    "plan_version",
                ),
                source_tp_rank=_coerce_optional_int(
                    data.get("source_tp_rank"),
                    "source_tp_rank",
                ),
                source_tp_size=_coerce_positive_int(
                    data.get("source_tp_size", 1),
                    "source_tp_size",
                ),
                target_tp_rank=_coerce_optional_int(
                    data.get("target_tp_rank"),
                    "target_tp_rank",
                ),
                target_tp_size=_coerce_positive_int(
                    data.get("target_tp_size", 1),
                    "target_tp_size",
                ),
            )
            for rank, size, name in (
                (
                    plan.source_tp_rank,
                    plan.source_tp_size,
                    "source_tp_rank",
                ),
                (
                    plan.target_tp_rank,
                    plan.target_tp_size,
                    "target_tp_rank",
                ),
            ):
                if rank is not None and (rank < 0 or rank >= size):
                    raise ValueError(f"{name} must be in [0, {size})")
            return plan
        except KeyError as err:
            raise ValueError(f"SharedHiCache plan missing {err.args[0]}") from err

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["router_block_hashes"] = list(self.router_block_hashes)
        value["engine_block_hashes"] = list(self.engine_block_hashes)
        return value

    @property
    def planned_router_block_hashes(self) -> tuple[int, ...]:
        return self.router_block_hashes[: self.planned_prefix_blocks]

    @property
    def planned_engine_block_hashes(self) -> tuple[int, ...]:
        return self.engine_block_hashes[: self.planned_prefix_blocks]

    def is_shared_hicache(self) -> bool:
        return self.source_medium == SHARED_HICACHE_SOURCE_MEDIUM

    def is_expired(self, now_ms: Optional[int] = None) -> bool:
        return self.expires_at_ms <= (now_ms if now_ms is not None else _now_ms())
