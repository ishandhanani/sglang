from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

import zmq

from sglang.srt.environ import envs
from sglang.srt.observability.req_time_stats import real_time
from sglang.srt.platforms import current_platform

if TYPE_CHECKING:
    from sglang.srt.rust_server.server import RustServer


class IdleSleeper:
    """
    In setups which have long inactivity periods it is desirable to reduce
    system power consumption when sglang does nothing. This would lead not only
    to power savings, but also to more CPU thermal headroom when a request
    eventually comes. This is important in cases when multiple GPUs are connected
    as each GPU would otherwise pin one thread at 100% CPU usage.

    The simplest solution is to use zmq.Poller on all sockets that may receive
    data that needs handling immediately.
    """

    def __init__(self, sockets, can_empty_cache=None):
        self.poller = zmq.Poller()
        self.last_empty_time = real_time()
        self.can_empty_cache = can_empty_cache
        for s in sockets:
            self.poller.register(s, zmq.POLLIN)

        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()

    def maybe_sleep(self):
        self.poller.poll(1000)
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            if self.can_empty_cache is None or self.can_empty_cache():
                current_platform.empty_cache()


# With a hint socket bound, the Rust-server park is cut into slices of this
# length and the socket is polled between them (see RustServerIdleSleeper).
HINT_POLL_MS = 20


class RustServerIdleSleeper:
    """Idle sleeper for the embedded Rust server.

    The Rust ingress is an in-process request ring, not a zmq socket.
    Instead park directly on the ring: ``wait_request`` blocks until
    a request is pushed — the request ring wakes the parked thread
    the instant a producer pushes, so there's no added latency for real
    requests — or the timeout elapses.

    The --hint-zmq-endpoint socket is a zmq socket the ring's Rust-side
    select cannot watch, and the scheduler thread owns the socket so no other
    thread may poll it. With a hint socket bound the park is therefore cut
    into ``hint_poll_ms`` slices of ``wait_request`` with a zero-timeout poll
    of the socket between them: a request still wakes the park at once, and
    a hint waits at most one slice (HINT_POLL_MS, 20 ms by default) instead
    of the full ``timeout_ms``.
    """

    def __init__(
        self,
        rust_server: RustServer,
        timeout_ms: int = 1000,
        can_empty_cache=None,
        hint_socket: Optional[zmq.Socket] = None,
        hint_poll_ms: int = HINT_POLL_MS,
    ):
        self.rust_server = rust_server
        self.timeout_ms = timeout_ms
        self.last_empty_time = real_time()
        self.can_empty_cache = can_empty_cache
        self.empty_cache_interval = envs.SGLANG_EMPTY_CACHE_INTERVAL.get()
        self.hint_poll_ms = hint_poll_ms
        self.hint_poller = None
        if hint_socket is not None:
            self.hint_poller = zmq.Poller()
            self.hint_poller.register(hint_socket, zmq.POLLIN)

    def park(self) -> bool:
        """Block until the ring holds a request, the hint socket holds a
        message, or ``timeout_ms`` elapse. True when woken for work."""
        if self.hint_poller is None:
            return bool(self.rust_server.wait_request(self.timeout_ms))
        deadline = time.monotonic() + self.timeout_ms / 1000
        while True:
            if self.hint_poller.poll(0):
                return True
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                return False
            if self.rust_server.wait_request(min(self.hint_poll_ms, remaining_ms)):
                return True

    def maybe_sleep(self):
        self.park()
        if (
            self.empty_cache_interval > 0
            and real_time() - self.last_empty_time > self.empty_cache_interval
        ):
            self.last_empty_time = real_time()
            if self.can_empty_cache is None or self.can_empty_cache():
                current_platform.empty_cache()
