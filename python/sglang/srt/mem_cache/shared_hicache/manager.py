from __future__ import annotations

import atexit
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Mapping, Optional

from sglang.srt.environ import envs
from sglang.srt.mem_cache.shared_hicache.config import shared_hicache_timeout_secs
from sglang.srt.mem_cache.shared_hicache.control import (
    SHARED_HICACHE_TRANSFER_DONE,
    SHARED_HICACHE_TRANSFER_REQUEST,
    SharedHiCacheTargetTransferTracker,
)
from sglang.srt.mem_cache.shared_hicache.plan import SharedHiCachePlan
from sglang.srt.mem_cache.shared_hicache.route import (
    SharedHiCacheSourceRouteRegistry,
)
from sglang.srt.mem_cache.shared_hicache.service import SharedHiCacheSourceService
from sglang.srt.mem_cache.shared_hicache.source_queue import (
    SharedHiCacheSourceTransferQueue,
)
from sglang.srt.mem_cache.shared_hicache.target_side import (
    SharedHiCacheResult,
    SharedHiCacheTarget,
    SharedHiCacheTargetReuse,
    validate_shared_hicache_plan,
)
from sglang.srt.mem_cache.shared_hicache.topology import SharedHiCacheTopology
from sglang.srt.mem_cache.shared_hicache.transfer import (
    make_shared_hicache_transfer_backend,
)
from sglang.srt.mem_cache.shared_hicache.transfer.common import (
    SharedHiCacheTransferBackend,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class SharedHiCacheManager:
    def __init__(
        self,
        *,
        server_args: ServerArgs,
        tree_cache,
        worker_id: str,
        topology: SharedHiCacheTopology,
        direct_transfer: SharedHiCacheTransferBackend,
        metrics_collector=None,
    ):
        self.tree_cache = tree_cache
        self.worker_id = worker_id
        self._set_topology(topology)
        self.timeout_secs = shared_hicache_timeout_secs()
        self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy
        self.direct_transfer = direct_transfer
        self.metrics_collector = metrics_collector
        self.endpoint = self._local_control_endpoint(server_args)
        self.source_service: Optional[SharedHiCacheSourceService] = None
        self.source_route_registry = SharedHiCacheSourceRouteRegistry()
        self._shutdown = False

        fetch_worker_limit = max(
            1,
            int(envs.SGLANG_SHARED_HICACHE_FETCH_WORKERS.get()),
        )
        source_worker_limit = fetch_worker_limit
        self.target_cache = SharedHiCacheTarget(
            tree_cache=tree_cache,
            metrics_collector=metrics_collector,
        )
        self.target_transfer_tracker = SharedHiCacheTargetTransferTracker(
            transfer_backend=direct_transfer,
        )
        self.target_reuse = SharedHiCacheTargetReuse(
            tree_cache=tree_cache,
            worker_id=worker_id,
            topology=self.topology,
            direct_transfer=direct_transfer,
            target_cache=self.target_cache,
            target_transfer_tracker=self.target_transfer_tracker,
            endpoint=self.endpoint,
            send_control_message=self._send_control_message,
            source_control_endpoint_for_plan=self._source_control_endpoint_for_plan,
            timeout_secs=self.timeout_secs,
            prefetch_stop_policy=self.prefetch_stop_policy,
            fetch_worker_limit=fetch_worker_limit,
            metrics_collector=metrics_collector,
        )

        self._direct_transfer_shutdown_done = False
        self._direct_transfer_shutdown_deferred = False
        self._direct_transfer_shutdown_lock = threading.Lock()

        self.source_transfer_queue = SharedHiCacheSourceTransferQueue(
            tree_cache=tree_cache,
            worker_id=worker_id,
            transfer_backend=direct_transfer,
            worker_limit=source_worker_limit,
            send_transfer_done=self._send_transfer_done,
            topology=self.topology,
        )
        self.source_service = SharedHiCacheSourceService(
            endpoint=self.endpoint,
            worker_id=self.worker_id,
            handle_control_message=self._handle_control_message,
        )
        self.source_service.start()
        atexit.register(self.shutdown)

    def _set_topology(
        self,
        topology: SharedHiCacheTopology,
    ) -> None:
        self.topology = topology
        for key, value in self.topology.to_dict().items():
            setattr(self, key, value)

    def _local_control_endpoint(
        self,
        server_args: ServerArgs,
    ) -> str:
        bootstrap_port = server_args.shared_hicache_bootstrap_port
        port = int(bootstrap_port) + int(self.tp_rank)
        if port > 65535:
            raise ValueError(
                "shared_hicache_bootstrap_port + tp_rank exceeds 65535"
            )
        host = str(server_args.host).strip()
        if not host:
            raise ValueError("host must be non-empty when SharedHiCache is enabled")
        return f"tcp://{host}:{port}"

    @classmethod
    def from_scheduler(cls, scheduler) -> Optional["SharedHiCacheManager"]:
        server_args = scheduler.server_args
        if not server_args.enable_shared_hicache:
            return None
        if not scheduler.enable_hierarchical_cache:
            logger.warning(
                "SharedHiCache disabled because hierarchical cache is not enabled"
            )
            return None
        required_tree_methods = (
            "lookup_hicache_host_blocks",
            "insert_shared_hicache_device_blocks",
        )
        missing_tree_methods = [
            name
            for name in required_tree_methods
            if not callable(getattr(scheduler.tree_cache, name, None))
        ]
        if missing_tree_methods:
            logger.warning(
                "SharedHiCache disabled because the active tree cache lacks HiCache "
                "shared-cache primitives: %s",
                ", ".join(missing_tree_methods),
            )
            return None
        worker_id = server_args.shared_hicache_worker_id
        if worker_id is None:
            logger.warning(
                "SharedHiCache disabled because worker_id is not set; "
                "set --shared-hicache-worker-id"
            )
            return None
        topology = SharedHiCacheTopology.from_scheduler(scheduler)
        direct_transfer = make_shared_hicache_transfer_backend(
            scheduler, topology=topology
        )
        metrics_reporter = getattr(scheduler, "metrics_reporter", None)
        metrics_collector = (
            scheduler.metrics_collector
            if getattr(metrics_reporter, "enable_metrics", False)
            else None
        )
        return cls(
            server_args=server_args,
            tree_cache=scheduler.tree_cache,
            worker_id=worker_id,
            topology=topology,
            direct_transfer=direct_transfer,
            metrics_collector=metrics_collector,
        )

    def _shutdown_direct_transfer_backend(self) -> None:
        with self._direct_transfer_shutdown_lock:
            if self._direct_transfer_shutdown_done:
                return
            self.direct_transfer.shutdown()
            self._direct_transfer_shutdown_done = True

    def _defer_direct_transfer_shutdown(self) -> None:
        if self._direct_transfer_shutdown_deferred:
            return
        self._direct_transfer_shutdown_deferred = True

        def _wait_for_pending_and_shutdown():
            while self.has_pending():
                time.sleep(0.01)
            try:
                self.target_cache.release_quarantined_device_indices()
                self._shutdown_direct_transfer_backend()
            except Exception:
                logger.warning(
                    "SharedHiCache deferred direct transfer backend shutdown failed",
                    exc_info=True,
                )

        thread = threading.Thread(
            target=_wait_for_pending_and_shutdown,
            name="shared_hicache-direct-transfer-shutdown",
            daemon=True,
        )
        thread.start()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True

        source_service = self.source_service
        self.source_service = None
        if source_service is not None:
            source_service.shutdown()

        self.target_reuse.release_all_pending()

        timeout_secs = float(self.timeout_secs)
        deadline = time.monotonic() + min(max(timeout_secs, 0.0), 5.0)
        while self.has_pending() and time.monotonic() < deadline:
            time.sleep(0.01)

        if self.has_pending():
            logger.warning(
                "Deferring direct transfer backend shutdown while SharedHiCache work is still pending"
            )
            self.source_transfer_queue.shutdown(wait=False, cancel_futures=True)
            self._defer_direct_transfer_shutdown()
            return

        self.source_transfer_queue.shutdown(wait=False, cancel_futures=True)
        self.target_cache.release_quarantined_device_indices()
        self._shutdown_direct_transfer_backend()

    def _source_control_endpoint_for_plan(
        self,
        plan: SharedHiCachePlan,
    ) -> Optional[str]:
        source_tp_rank = (
            int(plan.source_tp_rank)
            if plan.source_tp_rank is not None
            else int(self.tp_rank)
        )
        source_control_endpoint = self.source_route_registry.resolve(
            plan.source_worker_id, source_tp_rank
        )
        if source_control_endpoint is None:
            logger.warning(
                "Shared HiCache source route unavailable plan_id=%s source_worker=%s source_tp_rank=%s",
                plan.plan_id,
                plan.source_worker_id,
                source_tp_rank,
            )
            return None
        return source_control_endpoint

    def _send_control_message(self, endpoint: str, payload: Mapping[str, Any]) -> None:
        source_service = self.source_service
        if source_service is None:
            raise RuntimeError("SharedHiCache ZMQ control service is not running")
        source_service.send(endpoint, payload)

    def _handle_control_message(self, payload: Mapping[str, Any]) -> None:
        kind = str(payload.get("kind") or "")
        if kind == SHARED_HICACHE_TRANSFER_REQUEST:
            response = self.source_transfer_queue.handle(payload)
            if not response.get("accepted"):
                target_endpoint = str(payload.get("target_control_endpoint") or "")
                if target_endpoint:
                    self._send_transfer_done(target_endpoint, response)
            return
        if kind == SHARED_HICACHE_TRANSFER_DONE:
            self.target_transfer_tracker.handle_done(payload)
            return
        logger.warning("Ignoring unknown SharedHiCache control message kind=%s", kind)

    def _send_transfer_done(self, endpoint: str, payload: Mapping[str, Any]) -> None:
        message = dict(payload)
        message["kind"] = SHARED_HICACHE_TRANSFER_DONE
        try:
            self._send_control_message(endpoint, message)
        except Exception:
            logger.warning(
                "SharedHiCache failed to send transfer completion endpoint=%s transfer_id=%s",
                endpoint,
                message.get("transfer_id"),
                exc_info=True,
            )

    def _validate_plan(self, plan: SharedHiCachePlan) -> Optional[str]:
        return validate_shared_hicache_plan(
            plan,
            worker_id=self.worker_id,
            topology=self.topology,
            page_size=self.tree_cache.page_size,
        )

    def has_pending(self) -> bool:
        source_transfer_count = self.source_transfer_queue.active_count()
        source_service = self.source_service
        active_source_count = (
            source_service.active_count() if source_service is not None else 0
        )
        return (
            self.target_reuse.has_pending()
            or source_transfer_count > 0
            or active_source_count > 0
        )

    def has_reuse_plan(self, req: Req) -> bool:
        return self.target_reuse.has_reuse_plan(req)

    def release_request(self, rid: str) -> None:
        self.target_reuse.release_request(rid)

    def prepare_reuse(self, req: Req) -> SharedHiCacheResult:
        self.source_route_registry.update(
            getattr(req, "shared_hicache_source_routes", ())
        )
        return self.target_reuse.prepare_reuse(req)
