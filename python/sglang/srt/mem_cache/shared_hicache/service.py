from __future__ import annotations

import json
import logging
import threading
from contextlib import suppress
from typing import Any, Callable, Mapping, Optional

import zmq

from sglang.srt.mem_cache.shared_hicache.control import endpoint_to_zmq
from sglang.srt.utils.network import config_socket

logger = logging.getLogger(__name__)


def _encode_control_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(payload), separators=(",", ":")).encode("utf-8")


def _decode_control_payload(frames: list[bytes]) -> Mapping[str, Any]:
    if len(frames) != 1:
        raise ValueError(
            f"expected one Shared HiCache control frame, got {len(frames)}"
        )
    payload = json.loads(frames[0].decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Shared HiCache control payload must be a JSON object")
    return payload


class SharedHiCacheSourceService:
    """ZMQ control plane matching disagg's push/pull transfer metadata path."""

    def __init__(
        self,
        *,
        endpoint: str,
        worker_id: Optional[str],
        handle_control_message: Callable[[Mapping[str, Any]], None],
    ):
        self.endpoint = endpoint
        self.worker_id = worker_id
        self.handle_control_message = handle_control_message
        self._context = zmq.Context()
        self._pull_socket = None
        self._poller = None
        self._thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()
        self._send_lock = threading.Lock()
        self._send_sockets: dict[str, zmq.Socket] = {}
        self._activity_lock = threading.Lock()
        self._active_ops = 0

    def start(self) -> None:
        endpoint = endpoint_to_zmq(self.endpoint)
        socket = self._context.socket(zmq.PULL)
        config_socket(socket, zmq.PULL)
        socket.bind(endpoint)
        self._pull_socket = socket
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        self._poller = poller
        self._thread = threading.Thread(
            target=self._run,
            name=f"shared_hicache-zmq-{endpoint}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Shared HiCache ZMQ control listening on %s for worker_id=%s",
            endpoint,
            self.worker_id,
        )

    def send(self, endpoint: str, payload: Mapping[str, Any]) -> None:
        endpoint = endpoint_to_zmq(endpoint)
        with self._send_lock:
            socket = self._send_sockets.get(endpoint)
            if socket is None:
                socket = self._context.socket(zmq.PUSH)
                config_socket(socket, zmq.PUSH)
                socket.connect(endpoint)
                self._send_sockets[endpoint] = socket
            socket.send_multipart([_encode_control_payload(payload)])

    def active_count(self) -> int:
        with self._activity_lock:
            return int(self._active_ops)

    def shutdown(self) -> None:
        self._shutdown.set()
        poller = self._poller
        socket = self._pull_socket
        if poller is not None and socket is not None:
            with suppress(Exception):
                poller.unregister(socket)
        self._poller = None
        self._pull_socket = None
        if socket is not None:
            try:
                socket.close(linger=0)
            except Exception:
                logger.debug("Shared HiCache ZMQ pull close failed", exc_info=True)
        with self._send_lock:
            sockets = list(self._send_sockets.values())
            self._send_sockets.clear()
        for send_socket in sockets:
            try:
                send_socket.close(linger=0)
            except Exception:
                logger.debug("Shared HiCache ZMQ push close failed", exc_info=True)
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=1)
            except Exception:
                logger.debug("Shared HiCache ZMQ control join failed", exc_info=True)
        try:
            self._context.term()
        except Exception:
            logger.debug("Shared HiCache ZMQ context termination failed", exc_info=True)

    def _run(self) -> None:
        while not self._shutdown.is_set():
            socket = self._pull_socket
            poller = self._poller
            if socket is None or poller is None:
                break
            try:
                events = dict(poller.poll(100))
            except zmq.ZMQError:
                if self._shutdown.is_set():
                    break
                logger.debug("Shared HiCache ZMQ poll failed", exc_info=True)
                continue
            if socket not in events:
                continue
            with self._activity_lock:
                self._active_ops += 1
            try:
                payload = _decode_control_payload(socket.recv_multipart())
                self.handle_control_message(payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                logger.warning("Ignoring malformed Shared HiCache ZMQ payload")
            except zmq.ZMQError:
                if not self._shutdown.is_set():
                    logger.debug("Shared HiCache ZMQ receive failed", exc_info=True)
            except Exception:
                logger.exception("Shared HiCache ZMQ control handler failed")
            finally:
                with self._activity_lock:
                    self._active_ops -= 1
