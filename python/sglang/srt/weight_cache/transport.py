# SPDX-License-Identifier: Apache-2.0
"""Pluggable tensor transport backends for weight_cache."""

from __future__ import annotations

import array
import logging
import os
import socket
import struct
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from sglang.srt.utils import MultiprocessingSerializer

from .protocol import send_msg
from .vmm_arena import (
    DEFAULT_ARENA_BYTES,
    ArenaExporter,
    ArenaImporter,
    device_supports_posix_fd,
    vmm_fd_available,
)

logger = logging.getLogger(__name__)

TORCH_IPC_BACKEND = "torch_ipc"
VMM_FD_BACKEND = "vmm_fd"

_FD_INDEX_STRUCT = struct.Struct("<Q")


def _send_fd(sock: socket.socket, fd: int, index: int) -> None:
    payload = _FD_INDEX_STRUCT.pack(index)
    fds = array.array("i", [int(fd)])
    sent = sock.sendmsg(
        [payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())]
    )
    if sent != len(payload):
        raise RuntimeError(f"sendmsg sent {sent} bytes, expected {len(payload)}")


def _recv_fd(sock: socket.socket) -> Tuple[int, int]:
    fd_item_size = array.array("i").itemsize
    data, ancdata, _, _ = sock.recvmsg(
        _FD_INDEX_STRUCT.size, socket.CMSG_SPACE(fd_item_size)
    )
    if len(data) != _FD_INDEX_STRUCT.size:
        raise RuntimeError(
            f"received truncated fd header: {len(data)} < {_FD_INDEX_STRUCT.size}"
        )
    index = _FD_INDEX_STRUCT.unpack(data)[0]
    fds = array.array("i")
    for level, cmsg_type, cmsg_data in ancdata:
        if level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fd_item_size)])
    if len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise RuntimeError(f"expected one fd, got {len(fds)}")
    return int(index), int(fds[0])


class WeightCacheTransportBackend(ABC):
    name: str

    def bind_device(self, device_id: int) -> None:
        """Client side: the CUDA device the imported tensors belong to (before receiving)."""
        return None

    @abstractmethod
    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare daemon-side entries for all tensors."""

    @abstractmethod
    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
        preloaded_weights_bytes: int = 0,
    ) -> None:
        """Send a successful fetch_state response."""

    @abstractmethod
    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Client-side receive hook after recv_msg."""

    @abstractmethod
    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        """Import a single tensor from one entry."""


class TorchIpcTransportBackend(WeightCacheTransportBackend):
    name = TORCH_IPC_BACKEND

    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        for name, (tensor, is_param) in state_tensors.items():
            entries[name] = {
                "handle": MultiprocessingSerializer.serialize(
                    tensor.data, output_str=True
                ),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "is_param": is_param,
            }
        return entries

    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
        preloaded_weights_bytes: int = 0,
    ) -> None:
        send_msg(
            conn,
            {
                "status": "ok",
                "config": config,
                "entries": entries,
                "pid": pid,
                "transport_backend": self.name,
                "preloaded_weights_bytes": preloaded_weights_bytes,
            },
        )

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        return result

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        return MultiprocessingSerializer.deserialize(entry["handle"])


class VmmFdTransportBackend(WeightCacheTransportBackend):
    """CUDA VMM arenas exported as POSIX file descriptors (see ``vmm_arena``).

    The daemon copies every exported storage into a few large shareable
    allocations and sends one fd per arena after the pickled response; the
    client imports and maps them, then views each tensor at its offset. Imported
    mappings outlive the daemon, so the client needs no PID watchdog and no
    shared PID or IPC namespace with it.
    """

    name = VMM_FD_BACKEND

    def __init__(
        self, *, device_id: Optional[int] = None, arena_bytes: int = DEFAULT_ARENA_BYTES
    ):
        self._device_id = device_id
        self._arena_bytes = arena_bytes
        self._exporter: Optional[ArenaExporter] = None
        self._importer: Optional[ArenaImporter] = None

    @classmethod
    def can_export_state(
        cls, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> bool:
        if not vmm_fd_available():
            return False
        devices = {t.device for t, _ in state_tensors.values()}
        if not devices or any(d.type != "cuda" for d in devices):
            return False
        return all(
            device_supports_posix_fd(
                torch.cuda.current_device() if d.index is None else d.index
            )
            for d in devices
        )

    def bind_device(self, device_id: int) -> None:
        self._device_id = int(device_id)

    def _device(self) -> int:
        return (
            torch.cuda.current_device() if self._device_id is None else self._device_id
        )

    @property
    def daemon_views(self) -> Dict[str, torch.Tensor]:
        """Daemon-side tensors over the arena copies, by name; empty before export."""
        return dict(self._exporter.views) if self._exporter is not None else {}

    @property
    def resident_bytes(self) -> int:
        """Device bytes the daemon's arenas occupy after export."""
        return self._exporter.resident_bytes if self._exporter is not None else 0

    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        self._exporter = ArenaExporter(self._device(), arena_bytes=self._arena_bytes)
        return self._exporter.export(state_tensors)

    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
        preloaded_weights_bytes: int = 0,
    ) -> None:
        if self._exporter is None:
            raise RuntimeError(
                "vmm_fd: send_fetch_state_response before prepare_export"
            )
        send_msg(
            conn,
            {
                "status": "ok",
                "config": config,
                "entries": entries,
                "pid": pid,
                "transport_backend": self.name,
                "arenas": list(self._exporter.arena_sizes),
                "preloaded_weights_bytes": preloaded_weights_bytes,
            },
        )
        for index, fd in enumerate(self._exporter.fds):
            _send_fd(conn, fd, index)

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        sizes = result.get("arenas")
        if not isinstance(sizes, list) or not all(
            isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in sizes
        ):
            raise RuntimeError(f"vmm_fd: daemon sent invalid arena sizes {sizes!r}")
        fds: list = [None] * len(sizes)
        try:
            for _ in sizes:
                index, fd = _recv_fd(sock)
                if not 0 <= index < len(sizes) or fds[index] is not None:
                    os.close(fd)
                    raise RuntimeError(
                        f"vmm_fd: unexpected arena descriptor index {index}"
                    )
                fds[index] = fd
        except BaseException:
            for fd in fds:
                if fd is not None:
                    os.close(fd)
            raise
        self._importer = ArenaImporter(self._device())
        self._importer.import_arenas(fds, sizes)
        return result

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        if self._importer is None:
            raise RuntimeError("vmm_fd: import_tensor before recv_fetch_state_response")
        return self._importer.tensor(entry)

    def close(self) -> None:
        if self._importer is not None:
            self._importer.close()
            self._importer = None
        if self._exporter is not None:
            self._exporter.close()
            self._exporter = None


AUTO_TRANSPORT = "auto"
TRANSPORT_CHOICES = (AUTO_TRANSPORT, TORCH_IPC_BACKEND, VMM_FD_BACKEND)


def choose_daemon_transport_backend(
    state_tensors: Mapping[str, Tuple[torch.Tensor, bool]],
    requested: str = AUTO_TRANSPORT,
    *,
    device_id: Optional[int] = None,
) -> WeightCacheTransportBackend:
    """The daemon's transport for ``--weight-cache-transport``.

    ``auto`` stays on ``torch_ipc`` until ``vmm_fd`` has parity evidence for every
    allowlisted quantization; asking for ``vmm_fd`` where the device or the
    process cannot export POSIX-fd shareable memory is an error, not a fallback.
    """
    if requested == VMM_FD_BACKEND:
        if not VmmFdTransportBackend.can_export_state(state_tensors):
            raise RuntimeError(
                "--weight-cache-transport vmm_fd needs CUDA tensors, the cuda-python "
                "driver bindings, and a GPU that exports POSIX file descriptor handles"
            )
        backend: WeightCacheTransportBackend = VmmFdTransportBackend(
            device_id=device_id
        )
    elif requested in (AUTO_TRANSPORT, TORCH_IPC_BACKEND):
        backend = TorchIpcTransportBackend()
    else:
        raise ValueError(
            f"unknown weight cache transport {requested!r}; expected one of {TRANSPORT_CHOICES}"
        )
    logger.info("[weight_cache] Using transport backend: %s", backend.name)
    return backend


def get_client_transport_backend(name: Optional[str]) -> WeightCacheTransportBackend:
    if name in (None, "", TORCH_IPC_BACKEND):
        return TorchIpcTransportBackend()
    if name == VMM_FD_BACKEND:
        return VmmFdTransportBackend()
    raise RuntimeError(f"Unknown weight cache transport backend {name!r}")
