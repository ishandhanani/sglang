from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sglang.srt.environ import envs


class SharedHiCacheTransferBackendType(str, Enum):
    AUTO = "auto"
    NIXL = "nixl"


SHARED_HICACHE_TRANSFER_BACKEND_CHOICES = [
    backend.value for backend in SharedHiCacheTransferBackendType
]


@dataclass(frozen=True)
class SharedHiCacheConfig:
    control_host: str
    bootstrap_port: int
    timeout_secs: float
    transfer_backend: SharedHiCacheTransferBackendType

    def __post_init__(self) -> None:
        backend = self.transfer_backend
        if not isinstance(backend, SharedHiCacheTransferBackendType):
            backend = SharedHiCacheTransferBackendType(str(backend).lower())
        object.__setattr__(self, "transfer_backend", backend)


def shared_hicache_transfer_backend_name(server_args, default: str = "auto") -> str:
    return str(
        getattr(server_args, "shared_hicache_transfer_backend", None) or default
    ).lower()


def shared_hicache_timeout_secs() -> float:
    timeout_secs = float(envs.SGLANG_SHARED_HICACHE_TIMEOUT_SECS.get())
    if timeout_secs <= 0:
        raise ValueError("SGLANG_SHARED_HICACHE_TIMEOUT_SECS must be > 0")
    return timeout_secs


def _normalize_control_host(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalize_port(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer port")
    if value <= 0 or value > 65535:
        raise ValueError(f"{field_name} must be in [1, 65535]")
    return int(value)


def _normalize_worker_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("shared_hicache_worker_id must be a non-empty string")
    return value.strip()


def normalize_shared_hicache_server_config(
    *,
    enable_shared_hicache: bool,
    worker_id: Optional[str],
    host: str,
    bootstrap_port: Optional[int],
    transfer_backend: str,
    enable_hierarchical_cache: bool,
) -> tuple[bool, Optional[str], Optional[SharedHiCacheConfig]]:
    if not enable_shared_hicache:
        return False, worker_id, None

    if not enable_hierarchical_cache:
        raise ValueError("--enable-shared-hicache requires --enable-hierarchical-cache")
    worker_id = _normalize_worker_id(worker_id)

    if bootstrap_port is None:
        raise ValueError("--enable-shared-hicache requires --shared-hicache-bootstrap-port")
    bootstrap_port = _normalize_port(
        bootstrap_port, "shared_hicache_bootstrap_port"
    )

    transfer_backend_name = str(transfer_backend or "auto").lower()
    if transfer_backend_name not in SHARED_HICACHE_TRANSFER_BACKEND_CHOICES:
        raise ValueError(
            "shared_hicache_transfer_backend must be one of "
            f"{SHARED_HICACHE_TRANSFER_BACKEND_CHOICES}, got {transfer_backend_name!r}"
        )
    transfer_backend = SharedHiCacheTransferBackendType(transfer_backend_name)

    return (
        True,
        worker_id,
        SharedHiCacheConfig(
            control_host=_normalize_control_host(
                host,
                "host",
            ),
            bootstrap_port=bootstrap_port,
            timeout_secs=shared_hicache_timeout_secs(),
            transfer_backend=transfer_backend,
        ),
    )
