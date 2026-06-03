from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Mapping, Optional

from sglang.srt.mem_cache.shared_hicache.source import (
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
        self._executor = ThreadPoolExecutor(
            max_workers=worker_limit,
            thread_name_prefix=f"shared_hicache-source-xfer-tp{self.topology.tp_rank}",
        )
        self._capacity = threading.BoundedSemaphore(max(8, worker_limit * 2))
        self._lock = threading.Lock()
        self._transfers: dict[str, Optional[Future]] = {}

    def active_count(self) -> int:
        with self._lock:
            return sum(
                1
                for future in self._transfers.values()
                if future is not None and not future.done()
            )

    def shutdown(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def handle(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        transfer_id = str(payload.get("transfer_id") or uuid.uuid4().hex)
        target_endpoint = str(payload.get("target_control_endpoint") or "")
        payload = dict(payload)
        payload["transfer_id"] = transfer_id
        request, error = parse_source_transfer_request(
            payload=payload,
            transfer_backend=self.transfer_backend,
            tree_cache=self.tree_cache,
        )
        if error is not None:
            response = dict(error)
            response.setdefault("ok", False)
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
            if transfer_id in self._transfers:
                self._capacity.release()
                return {
                    "ok": False,
                    "reason": "duplicate_transfer_id",
                    "transfer_id": transfer_id,
                    "transferred_blocks": 0,
                    "block_size_tokens": self.tree_cache.page_size,
                }
            self._transfers[transfer_id] = None
        try:
            future = self._executor.submit(
                execute_source_transfer_request,
                request=request,
                transfer_backend=self.transfer_backend,
                tree_cache=self.tree_cache,
                worker_id=self.worker_id,
                topology=self.topology,
            )
        except Exception:
            with self._lock:
                if self._transfers.get(transfer_id) is None:
                    self._transfers.pop(transfer_id, None)
            self._capacity.release()
            raise

        def _complete_source_transfer(done_future: Future) -> None:
            response: Mapping[str, Any]
            try:
                response = dict(done_future.result())
            except Exception:
                logger.exception(
                    "SharedHiCache source transfer job failed transfer_id=%s",
                    transfer_id,
                )
                response = {
                    "ok": False,
                    "reason": "source_transfer_exception",
                    "transferred_blocks": 0,
                    "block_size_tokens": self.tree_cache.page_size,
                }
            response = dict(response)
            response["transfer_id"] = transfer_id
            self.send_transfer_done(
                target_endpoint or request.target_control_endpoint, response
            )
            self._capacity.release()
            with self._lock:
                if self._transfers.get(transfer_id) is done_future:
                    self._transfers.pop(transfer_id, None)

        with self._lock:
            self._transfers[transfer_id] = future
        future.add_done_callback(_complete_source_transfer)

        return {
            "ok": True,
            "accepted": True,
            "pending": True,
            "reason": "accepted",
            "transfer_id": transfer_id,
            "block_size_tokens": self.tree_cache.page_size,
        }
