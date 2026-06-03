from __future__ import annotations

import json
import logging
import threading
import asyncio
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse

import requests
import zmq
from aiohttp import web

from sglang.srt.mem_cache.shared_hicache.control import endpoint_to_zmq
from sglang.srt.mem_cache.shared_hicache.plan import normalize_endpoint
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


def normalize_registry_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        return endpoint
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
        raise ValueError("shared HiCache registry endpoint must be http://host:port")
    return endpoint.rstrip("/")


def _registry_route_url(registry_endpoint: str) -> str:
    return f"{normalize_registry_endpoint(registry_endpoint)}/route"


def _registry_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    raise ValueError(f"{field_name} must be an integer")


class SharedHiCacheRegistryClient:
    """Disagg-style HTTP route lookup for concrete source rank endpoints."""

    def __init__(self, endpoint: str, *, timeout_secs: float = 1.0):
        self.endpoint = normalize_registry_endpoint(endpoint)
        self.timeout_secs = float(timeout_secs)

    def register(
        self,
        *,
        worker_id: int,
        tp_rank: int,
        tp_size: int,
        endpoint: str,
    ) -> None:
        response = requests.put(
            _registry_route_url(self.endpoint),
            json={
                "worker_id": int(worker_id),
                "tp_rank": int(tp_rank),
                "tp_size": int(tp_size),
                "endpoint": normalize_endpoint(endpoint),
            },
            timeout=self.timeout_secs,
        )
        response.raise_for_status()

    def resolve(self, *, worker_id: int, tp_rank: int) -> Optional[str]:
        response = requests.get(
            _registry_route_url(self.endpoint),
            params={"worker_id": int(worker_id), "tp_rank": int(tp_rank)},
            timeout=self.timeout_secs,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        endpoint = payload.get("endpoint") if isinstance(payload, Mapping) else None
        if not isinstance(endpoint, str) or not endpoint.strip():
            return None
        return normalize_endpoint(endpoint)


class SharedHiCacheRegistryServer:
    """Minimal `/route` registry matching disagg's concrete endpoint discovery."""

    def __init__(self, endpoint: str):
        self.endpoint = normalize_registry_endpoint(endpoint)
        parsed = urlparse(self.endpoint)
        self.host = parsed.hostname
        self.port = parsed.port
        self._routes: dict[tuple[int, int], dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[web.AppRunner] = None
        self.app = web.Application()
        self.app.router.add_route("*", "/route", self._handle_route)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_server,
            name=f"shared_hicache-registry-{self.endpoint}",
            daemon=True,
        )
        self._thread.start()

    async def _handle_route(self, request: web.Request):
        if request.method == "PUT":
            return await self._handle_route_put(request)
        if request.method == "GET":
            return await self._handle_route_get(request)
        return web.Response(text="Method not allowed", status=405)

    async def _handle_route_put(self, request: web.Request):
        data = await request.json()
        worker_id = _registry_int(data.get("worker_id"), "worker_id")
        tp_rank = _registry_int(data.get("tp_rank"), "tp_rank")
        tp_size = _registry_int(data.get("tp_size", 1), "tp_size")
        endpoint = normalize_endpoint(str(data.get("endpoint", "")))
        if tp_rank < 0 or tp_size <= 0 or tp_rank >= tp_size:
            return web.Response(text="tp_rank must be in [0, tp_size)", status=400)
        if not endpoint:
            return web.Response(text="endpoint must be non-empty", status=400)
        route = {
            "worker_id": worker_id,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "endpoint": endpoint,
        }
        async with self._lock:
            self._routes[(worker_id, tp_rank)] = route
        return web.json_response(route)

    async def _handle_route_get(self, request: web.Request):
        worker_id = _registry_int(request.query.get("worker_id"), "worker_id")
        tp_rank = _registry_int(request.query.get("tp_rank"), "tp_rank")
        async with self._lock:
            route = self._routes.get((worker_id, tp_rank))
        if route is None:
            return web.Response(text="route not found", status=404)
        return web.json_response(route)

    def _run_server(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._runner = web.AppRunner(self.app, access_log=None)
            self._loop.run_until_complete(self._runner.setup())
            site = web.TCPSite(self._runner, host=self.host, port=self.port)
            self._loop.run_until_complete(site.start())
            logger.info("Shared HiCache registry listening on %s", self.endpoint)
            self._loop.run_forever()
        except Exception:
            logger.exception("Shared HiCache registry failed")
        finally:
            if self._runner is not None and self._loop is not None:
                self._loop.run_until_complete(self._runner.cleanup())
            if self._loop is not None:
                self._loop.close()

    def shutdown(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=1)
            except Exception:
                logger.debug("Shared HiCache registry join failed", exc_info=True)


class SharedHiCacheSourceService:
    """ZMQ control plane matching disagg's push/pull transfer metadata path."""

    def __init__(
        self,
        *,
        endpoint: str,
        worker_id: Optional[int],
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
            try:
                poller.unregister(socket)
            except Exception:
                pass
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
