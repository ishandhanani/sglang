from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.utils import block_hash_aliases

SHARED_HICACHE_PLAN_VERSION = 1
SHARED_HICACHE_DIRECT_TIMEOUT_REASON = "source_transfer_timeout_maybe_inflight"
SHARED_HICACHE_SOURCE_MEDIUM = StorageMedium.CPU.value


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


def _coerce_block_hash(value: Any) -> int:
    return _coerce_int(value, "block_hash")


def expand_block_hash_aliases(values: Iterable[int]) -> set[int]:
    aliases: set[int] = set()
    for value in values:
        aliases.update(block_hash_aliases(value))
    return aliases


@dataclass(frozen=True)
class SharedHiCachePlan:
    plan_id: str
    request_id: str
    target_worker_id: str
    source_worker_id: str
    source_host: str
    source_bootstrap_port: int
    source_medium: str
    block_hashes: tuple[int, ...]
    planned_prefix_blocks: int
    block_size_tokens: int
    created_at_ms: int
    expires_at_ms: int
    start_block_index: int = 0
    plan_version: int = SHARED_HICACHE_PLAN_VERSION
    kv_block_hashes: tuple[int, ...] = ()
    source_tp_rank: Optional[int] = None
    source_tp_size: int = 1
    target_tp_rank: Optional[int] = None
    target_tp_size: int = 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SharedHiCachePlan":
        if not isinstance(data, Mapping):
            raise ValueError("SharedHiCache plan must be a mapping")

        if "source_endpoint" in data:
            raise ValueError(
                "source_endpoint is not supported; use source_host/source_bootstrap_port"
            )

        if "block_hashes" not in data:
            raise ValueError("SharedHiCache plan missing block_hashes")
        block_hashes = tuple(
            _coerce_block_hash(item)
            for item in _coerce_array(data["block_hashes"], "block_hashes")
        )
        kv_block_hashes_raw = data.get("kv_block_hashes", ())
        if kv_block_hashes_raw is None:
            kv_block_hashes_raw = ()
        kv_block_hashes = tuple(
            _coerce_block_hash(item)
            for item in _coerce_array(kv_block_hashes_raw, "kv_block_hashes")
        )
        if kv_block_hashes and len(kv_block_hashes) != len(block_hashes):
            raise ValueError(
                "kv_block_hashes length must match block_hashes when provided"
            )

        planned_prefix_blocks = _coerce_int(
            data.get("planned_prefix_blocks", len(block_hashes)),
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
                block_hashes=block_hashes,
                planned_prefix_blocks=min(planned_prefix_blocks, len(block_hashes)),
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
                kv_block_hashes=kv_block_hashes,
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

    @classmethod
    def coerce(cls, data: Optional[Any]) -> Optional["SharedHiCachePlan"]:
        if data is None:
            return None
        if isinstance(data, cls):
            return data
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["block_hashes"] = list(self.block_hashes)
        value["kv_block_hashes"] = list(self.kv_block_hashes)
        return value

    @property
    def planned_hashes(self) -> tuple[int, ...]:
        return self.block_hashes[: self.planned_prefix_blocks]

    @property
    def planned_kv_block_hashes(self) -> tuple[int, ...]:
        return self.kv_block_hashes[: self.planned_prefix_blocks]

    def is_shared_hicache(self) -> bool:
        return self.source_medium == SHARED_HICACHE_SOURCE_MEDIUM

    def is_expired(self, now_ms: Optional[int] = None) -> bool:
        return self.expires_at_ms <= (now_ms if now_ms is not None else _now_ms())
