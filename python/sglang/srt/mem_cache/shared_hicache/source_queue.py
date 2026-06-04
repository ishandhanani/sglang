from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Mapping, Optional

from sglang.srt.mem_cache.shared_hicache.source import (
    SourceTransferRequest,
    execute_source_transfer_request,
    parse_source_transfer_request,
)
from sglang.srt.mem_cache.shared_hicache.topology import SharedHiCacheTopology
from sglang.srt.mem_cache.shared_hicache.transfer.common import (
    SharedHiCacheTransferBackend,
)

logger = logging.getLogger(__name__)


class SharedHiCacheSourceTransferQueue:
    """Owns source-side asynchronous transfer execution for one local TP rank."""

    def __init__(
        self,
        *,
        tree_cache,
        worker_id: Optional[str],
        transfer_backend: SharedHiCacheTransferBackend,
        worker_limit: int,
        send_transfer_done: Callable[[str, Mapping[str, Any]], None],
        topology: SharedHiCacheTopology,
    ):
        self.tree_cache = tree_cache
        self.worker_id = worker_id
        self.transfer_backend = transfer_backend
        self.send_transfer_done = send_transfer_done
        self.topology = topology

        worker_limit = max(1, int(worker_limit))
        self._jobs: queue.Queue[Optional[SourceTransferRequest]] = queue.Queue()
        self._capacity = threading.BoundedSemaphore(max(8, worker_limit * 2))
        self._lock = threading.Lock()
        self._transfers: set[str] = set()
        self._shutdown = False
        self._workers: list[threading.Thread] = []

        ready: queue.Queue[Optional[BaseException]] = queue.Queue()
        for index in range(worker_limit):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(index, ready),
                daemon=True,
                name=(
                    "shared_hicache-source-xfer-"
                    f"tp{self.topology.tp_rank}-{index}"
                ),
            )
            thread.start()
            self._workers.append(thread)

        for _ in self._workers:
            error = ready.get()
            if error is not None:
                self.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError(
                    "SharedHiCache source transfer worker failed to initialize"
                ) from error

    def _worker_loop(
        self,
        index: int,
        ready: queue.Queue[Optional[BaseException]],
    ) -> None:
        try:
            # NIXL source contexts are thread-owned; initialize before serving jobs.
            source_worker = self.transfer_backend.create_source_worker()
        except BaseException as err:
            ready.put(err)
            return

        logger.info(
            "SharedHiCache source transfer worker ready tp_rank=%d worker_index=%d",
            self.topology.tp_rank,
            index,
        )
        ready.put(None)

        while True:
            request = self._jobs.get()
            if request is None:
                self._jobs.task_done()
                return
            self._run_job(request, source_worker)
            self._jobs.task_done()

    def _run_job(self, request: SourceTransferRequest, source_worker) -> None:
        response: Mapping[str, Any]
        try:
            response = dict(
                execute_source_transfer_request(
                    request=request,
                    transfer_backend=source_worker,
                    tree_cache=self.tree_cache,
                    worker_id=self.worker_id,
                    topology=self.topology,
                )
            )
        except Exception:
            logger.exception(
                "SharedHiCache source transfer job failed transfer_id=%s",
                request.transfer_id,
            )
            response = {
                "ok": False,
                "reason": "source_transfer_exception",
                "transferred_blocks": 0,
                "block_size_tokens": self.tree_cache.page_size,
            }
        response["transfer_id"] = request.transfer_id
        try:
            self.send_transfer_done(request.target_control_endpoint, response)
        except Exception:
            logger.exception(
                "SharedHiCache source transfer completion send failed transfer_id=%s",
                request.transfer_id,
            )
        finally:
            self._capacity.release()
            with self._lock:
                self._transfers.discard(request.transfer_id)

    def active_count(self) -> int:
        with self._lock:
            return len(self._transfers)

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        with self._lock:
            self._shutdown = True

        if cancel_futures:
            while True:
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None:
                    with self._lock:
                        self._transfers.discard(job.transfer_id)
                    self._capacity.release()
                self._jobs.task_done()

        for _ in self._workers:
            self._jobs.put(None)

        if wait:
            for worker in self._workers:
                worker.join()

    def handle(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        transfer_id = str(payload.get("transfer_id") or "")
        if not transfer_id:
            return {
                "ok": False,
                "reason": "malformed_transfer_request:missing_transfer_id",
                "transferred_blocks": 0,
                "block_size_tokens": self.tree_cache.page_size,
            }
        payload = dict(payload)
        payload["transfer_id"] = transfer_id
        request, error = parse_source_transfer_request(
            payload=payload,
            transfer_backend=self.transfer_backend,
            tree_cache=self.tree_cache,
        )
        if error is not None:
            response = dict(error)
            response["transfer_id"] = transfer_id
            response["transferred_blocks"] = 0
            return response
        assert request is not None

        if not self._capacity.acquire(blocking=False):
            return {
                "ok": False,
                "reason": "source_transfer_queue_full",
                "transfer_id": transfer_id,
                "transferred_blocks": 0,
                "block_size_tokens": self.tree_cache.page_size,
            }
        with self._lock:
            if self._shutdown:
                self._capacity.release()
                return {
                    "ok": False,
                    "reason": "source_transfer_queue_shutdown",
                    "transfer_id": transfer_id,
                    "transferred_blocks": 0,
                    "block_size_tokens": self.tree_cache.page_size,
                }
            if transfer_id in self._transfers:
                self._capacity.release()
                return {
                    "ok": False,
                    "reason": "duplicate_transfer_id",
                    "transfer_id": transfer_id,
                    "transferred_blocks": 0,
                    "block_size_tokens": self.tree_cache.page_size,
                }
            self._transfers.add(transfer_id)

        self._jobs.put(request)

        return {
            "ok": True,
            "accepted": True,
            "pending": True,
            "reason": "accepted",
            "transfer_id": transfer_id,
            "block_size_tokens": self.tree_cache.page_size,
        }
