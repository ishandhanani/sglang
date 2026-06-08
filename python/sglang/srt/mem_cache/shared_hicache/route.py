from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterable, Mapping, Optional


@dataclass(frozen=True)
class SharedHiCacheSourceRoute:
    worker_id: str
    tp_rank: int
    endpoint: str

    @classmethod
    def from_endpoint_dict(cls, data: Mapping[str, Any]) -> "SharedHiCacheSourceRoute":
        worker_id = str(data.get("source_worker_id") or data.get("worker_id") or "")
        if not worker_id:
            raise ValueError("SharedHiCache source route missing source_worker_id")
        tp_rank = int(data.get("source_tp_rank", data.get("tp_rank")))
        endpoint = str(data["endpoint"]).strip().rstrip("/")
        if not endpoint:
            raise ValueError("SharedHiCache source route endpoint must be non-empty")
        return cls(worker_id=worker_id, tp_rank=tp_rank, endpoint=endpoint)


class SharedHiCacheSourceRouteRegistry:
    def __init__(self, routes: Iterable[SharedHiCacheSourceRoute] = ()):
        self._lock = Lock()
        self._routes: dict[tuple[str, int], str] = {}
        self.update(routes)

    def update(self, routes: Iterable[SharedHiCacheSourceRoute]) -> None:
        with self._lock:
            for route in routes:
                self._routes[(route.worker_id, int(route.tp_rank))] = route.endpoint

    def resolve(self, worker_id: str, tp_rank: int) -> Optional[str]:
        with self._lock:
            return self._routes.get((str(worker_id), int(tp_rank)))


def shared_hicache_source_routes_from_hint(
    hint: Any,
) -> tuple[SharedHiCacheSourceRoute, ...]:
    if hint is None:
        return ()
    if isinstance(hint, Mapping):
        entries = [hint]
    elif isinstance(hint, list):
        entries = hint
    else:
        raise ValueError("shared_hicache_source_routes must be a mapping or list")

    routes: list[SharedHiCacheSourceRoute] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("SharedHiCache source route must be a mapping")
        routes.append(SharedHiCacheSourceRoute.from_endpoint_dict(entry))

    return tuple(routes)
