from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from sglang.srt.disaggregation.kv_events import StorageMedium

SHARED_HICACHE_PLAN_VERSION = 1
SHARED_HICACHE_DIRECT_TIMEOUT_REASON = "source_transfer_timeout_maybe_inflight"
SHARED_HICACHE_SOURCE_MEDIUM = StorageMedium.CPU.value
_SIGNED_INT64_MAX = 2**63 - 1
_UINT64_MAX = 2**64 - 1


def _now_ms() -> int:
    return int(time.time() * 1000)


def _engine_hash_for_source_lookup(value: Any) -> int:
    value = int(value)
    if value > _SIGNED_INT64_MAX:
        if value > _UINT64_MAX:
            raise ValueError("engine_block_hashes must fit in uint64")
        # SGLang KV events and the source host index use signed int64 hashes.
        # Dynamo can still serialize that same bit pattern as u64 over JSON.
        value -= 2**64
    return value


@dataclass(frozen=True)
class SharedHiCachePlan:
    plan_id: str
    request_id: str
    target_worker_id: str
    source_worker_id: str
    source_medium: str
    router_block_hashes: tuple[int, ...]
    engine_block_hashes: tuple[int, ...]
    planned_prefix_blocks: int
    block_size_tokens: int
    created_at_ms: int
    expires_at_ms: int
    start_block_index: int
    plan_version: int
    source_tp_size: int
    target_tp_size: int
    x_request_id: Optional[str] = None
    source_tp_rank: Optional[int] = None
    target_tp_rank: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SharedHiCachePlan":
        if not isinstance(data, Mapping):
            raise ValueError("SharedHiCache plan must be a mapping")

        try:
            router_block_hashes = tuple(int(item) for item in data["router_block_hashes"])
        except KeyError as err:
            raise ValueError("SharedHiCache plan missing router_block_hashes") from err
        except (TypeError, ValueError) as err:
            raise ValueError("router_block_hashes must be an integer array") from err
        # These arrays are parallel but intentionally separate today:
        # - router_block_hashes are router/request block identities. They
        #   preserve plan order and label the pages returned to the target.
        # - engine_block_hashes are source-worker HostPinned lookup keys from
        #   SGLang KV events. The source host index is keyed by these values.
        # If Dynamo and SGLang later share one canonical block-hash contract
        # (same representation, algorithm, and parent-chaining semantics), this
        # source-lookup field can collapse into router_block_hashes.
        try:
            engine_block_hashes = tuple(
                _engine_hash_for_source_lookup(item)
                for item in data["engine_block_hashes"]
            )
        except KeyError as err:
            raise ValueError("SharedHiCache plan missing engine_block_hashes") from err
        except (TypeError, ValueError) as err:
            raise ValueError("engine_block_hashes must be an integer array") from err
        if len(engine_block_hashes) != len(router_block_hashes):
            raise ValueError(
                "engine_block_hashes length must match router_block_hashes"
            )

        try:
            planned_prefix_blocks = int(data["planned_prefix_blocks"])
            start_block_index = int(data["start_block_index"])
            source_tp_rank = data.get("source_tp_rank")
            target_tp_rank = data.get("target_tp_rank")
            if planned_prefix_blocks < 0:
                raise ValueError("planned_prefix_blocks must be non-negative")
            if planned_prefix_blocks > len(router_block_hashes):
                raise ValueError(
                    "planned_prefix_blocks must not exceed router_block_hashes length"
                )
            if start_block_index < 0:
                raise ValueError("start_block_index must be non-negative")
            return cls(
                plan_id=str(data["plan_id"]),
                request_id=str(data["request_id"]),
                x_request_id=(
                    None
                    if data.get("x_request_id") is None
                    else str(data.get("x_request_id"))
                ),
                target_worker_id=str(data["target_worker_id"]),
                source_worker_id=str(data["source_worker_id"]),
                source_medium=str(data["source_medium"]),
                router_block_hashes=router_block_hashes,
                engine_block_hashes=engine_block_hashes,
                planned_prefix_blocks=planned_prefix_blocks,
                block_size_tokens=int(data["block_size_tokens"]),
                created_at_ms=int(data["created_at_ms"]),
                expires_at_ms=int(data["expires_at_ms"]),
                start_block_index=start_block_index,
                plan_version=int(data["plan_version"]),
                source_tp_rank=None if source_tp_rank is None else int(source_tp_rank),
                source_tp_size=int(data["source_tp_size"]),
                target_tp_rank=None if target_tp_rank is None else int(target_tp_rank),
                target_tp_size=int(data["target_tp_size"]),
            )
        except KeyError as err:
            raise ValueError(f"SharedHiCache plan missing {err.args[0]}") from err

    def to_dict(self) -> dict[str, Any]:
        data = {
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "target_worker_id": self.target_worker_id,
            "source_worker_id": self.source_worker_id,
            "source_medium": self.source_medium,
            "router_block_hashes": list(self.router_block_hashes),
            "engine_block_hashes": list(self.engine_block_hashes),
            "planned_prefix_blocks": self.planned_prefix_blocks,
            "block_size_tokens": self.block_size_tokens,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "start_block_index": self.start_block_index,
            "plan_version": self.plan_version,
            "source_tp_rank": self.source_tp_rank,
            "source_tp_size": self.source_tp_size,
            "target_tp_rank": self.target_tp_rank,
            "target_tp_size": self.target_tp_size,
        }
        if self.x_request_id is not None:
            data["x_request_id"] = self.x_request_id
        return data

    @property
    def planned_router_block_hashes(self) -> tuple[int, ...]:
        return self.router_block_hashes[: self.planned_prefix_blocks]

    @property
    def planned_engine_block_hashes(self) -> tuple[int, ...]:
        return self.engine_block_hashes[: self.planned_prefix_blocks]

    def is_expired(self, now_ms: Optional[int] = None) -> bool:
        return self.expires_at_ms <= (now_ms if now_ms is not None else _now_ms())
