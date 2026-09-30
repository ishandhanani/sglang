# SPDX-License-Identifier: Apache-2.0
"""CUDA VMM arenas behind the weight cache ``vmm_fd`` transport.

Daemon side: every storage behind an exported tensor is copied into one of a
few large ``cuMemCreate`` allocations (arenas) created shareable as POSIX file
descriptors, the daemon's own tensors are rebound onto the copies so the
originals can be freed, and one fd per arena is handed to each client over
``SCM_RIGHTS``. Client side: each fd is imported with
``cuMemImportFromShareableHandle``, mapped into a fresh VA reservation, and
every tensor becomes a view at its recorded offset.

Physical memory stays alive while any process holds a mapping or an imported
handle, so a client outlives the daemon. The legacy CUDA IPC transport lacks
that property (its mappings dangle when the exporter exits, hence its SIGKILL
watchdog), which is why this transport exists (RFC #27310 section 4.2 phase 1,
roadmap #33522). Tensors keep their exact shape, stride and storage offset, and
tensors that share a storage in the daemon share it in the client too.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from sglang.srt.utils.cuda_vmm_utils import (
    VmmReservation,
    align_up,
    check_drv,
    get_device_granularity,
    make_device_allocation_prop,
    tensor_from_pointer,
)

try:
    from cuda.bindings import driver as _drv
except ImportError:  # pragma: no cover - exercised on hosts without cuda-python
    _drv = None

logger = logging.getLogger(__name__)

# Arenas are large so a model with many small tensors (MoE experts) does not pay
# the allocation granularity (2 MiB) per tensor.
DEFAULT_ARENA_BYTES = 1 << 30
# Byte alignment of every storage inside an arena: torch's allocator guarantees
# 512-byte alignment and kernels may assume it.
TENSOR_ALIGNMENT = 512


def vmm_fd_available() -> bool:
    """Whether this process can allocate and import POSIX-fd shareable VMM memory."""
    return _drv is not None and torch.cuda.is_available()


def _posix_fd_handle_type():
    return _drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR


def device_supports_posix_fd(device_id: int) -> bool:
    """Whether ``device_id`` can export allocations as POSIX file descriptors."""
    if not vmm_fd_available():
        return False
    device = check_drv(_drv.cuDeviceGet(int(device_id)), "cuDeviceGet")
    attr = _drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED
    return bool(
        check_drv(_drv.cuDeviceGetAttribute(attr, device), "cuDeviceGetAttribute")
    )


def plan_arenas(
    sizes: Sequence[int],
    *,
    arena_bytes: int,
    granularity: int,
    alignment: int = TENSOR_ALIGNMENT,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """First-fit placement of ``sizes`` (bytes) into arenas.

    Returns ``(arena_sizes, placements)`` with ``placements[i] = (arena, offset)``
    for storage ``i``. Storages are placed in the given order at ``alignment``
    aligned offsets; one that does not fit any arena's remaining space opens a
    new one, and one larger than ``arena_bytes`` gets an arena of its own. Every
    arena size is rounded up to the device granularity, the unit ``cuMemCreate``
    allocates in, so the sizes here are exactly what a client reserves and maps.
    """
    if arena_bytes <= 0 or granularity <= 0 or alignment <= 0:
        raise ValueError("arena_bytes, granularity and alignment must be positive")
    cursors: List[int] = []
    placements: List[Tuple[int, int]] = []
    for size in sizes:
        size = int(size)
        if size < 0:
            raise ValueError(f"negative storage size {size}")
        for index, cursor in enumerate(cursors):
            start = align_up(cursor, alignment)
            if start + size <= arena_bytes:
                placements.append((index, start))
                cursors[index] = start + size
                break
        else:
            placements.append((len(cursors), 0))
            cursors.append(size)
    arena_sizes = [align_up(max(cursor, 1), granularity) for cursor in cursors]
    return arena_sizes, placements


def _torch_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown tensor dtype {name!r} in weight cache entry")
    return dtype


def view_tensor(base: int, entry: Mapping[str, Any], device_id: int) -> torch.Tensor:
    """A tensor over the storage at ``base`` with the entry's shape, stride and offset."""
    dtype = _torch_dtype(entry["dtype"])
    storage_bytes = int(entry["storage_bytes"])
    element_size = torch.empty((), dtype=dtype).element_size()
    flat = tensor_from_pointer(
        int(base),
        storage_bytes,
        shape=(storage_bytes // element_size,),
        dtype=dtype,
        device_id=int(device_id),
    )
    return flat.as_strided(
        tuple(entry["shape"]), tuple(entry["stride"]), int(entry["storage_offset"])
    )


class ArenaExporter:
    """Daemon side: copy exported storages into shareable arenas and export fds.

    After :meth:`export`, ``entries`` describes every tensor for the wire,
    ``views`` holds a tensor per name over the arena copy (the daemon rebinds its
    model onto these so the original allocations can be freed), ``arena_sizes``
    lists the bytes a client must map per arena, and ``fds`` are the exported
    descriptors, one per arena, owned by the daemon for its lifetime.
    """

    def __init__(self, device_id: int, *, arena_bytes: int = DEFAULT_ARENA_BYTES):
        if not vmm_fd_available():
            raise RuntimeError(
                "the vmm_fd weight cache transport needs CUDA and the cuda-python "
                "driver bindings (cuda.bindings.driver)"
            )
        self.device_id = int(device_id)
        self.arena_bytes = int(arena_bytes)
        self.arena_sizes: List[int] = []
        self.fds: List[int] = []
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.views: Dict[str, torch.Tensor] = {}
        self._reservations: List[VmmReservation] = []
        self._handles: List[int] = []

    @property
    def resident_bytes(self) -> int:
        """Bytes the arenas occupy on the device (the exported weights plus padding)."""
        return sum(self.arena_sizes)

    def export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        if self._reservations:
            raise RuntimeError("ArenaExporter.export called twice")
        torch.cuda.set_device(self.device_id)

        # One copy per storage: tensors sharing a storage (tied weights, views into
        # a merged buffer) keep sharing it on the client.
        storage_index: Dict[Tuple[int, int], int] = {}
        storages: List[Tuple[int, int]] = []
        refs: List[Tuple[str, int, torch.Tensor, bool]] = []
        for name, (tensor, is_param) in state_tensors.items():
            if not tensor.is_cuda:
                raise RuntimeError(
                    f"[vmm_fd] tensor {name!r} is on {tensor.device}, not on the daemon's GPU"
                )
            storage = tensor.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
            index = storage_index.setdefault(key, len(storages))
            if index == len(storages):
                storages.append(key)
            refs.append((name, index, tensor, is_param))

        granularity = get_device_granularity(self.device_id)
        prop = make_device_allocation_prop(
            self.device_id, handle_types=int(_posix_fd_handle_type())
        )
        sizes = [nbytes for _, nbytes in storages]
        self.arena_sizes, placements = plan_arenas(
            sizes, arena_bytes=self.arena_bytes, granularity=granularity
        )
        for size in self.arena_sizes:
            reservation = VmmReservation(
                size, prop, self.device_id, alignment=granularity
            )
            handle = reservation.map(0, size, retain_handle=True)
            self._reservations.append(reservation)
            self._handles.append(int(handle))

        for (pointer, nbytes), (arena, offset) in zip(storages, placements):
            if nbytes:
                check_drv(
                    _drv.cuMemcpyDtoD(
                        self._reservations[arena].base + offset, pointer, nbytes
                    ),
                    "cuMemcpyDtoD(arena)",
                )
        torch.cuda.synchronize(self.device_id)

        for name, index, tensor, is_param in refs:
            arena, offset = placements[index]
            entry = {
                "arena": arena,
                "offset": offset,
                "storage_bytes": sizes[index],
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "storage_offset": int(tensor.storage_offset()),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "is_param": bool(is_param),
            }
            self.entries[name] = entry
            self.views[name] = view_tensor(
                self._reservations[arena].base + offset, entry, self.device_id
            )

        posix_fd = _posix_fd_handle_type()
        for handle in self._handles:
            fd = check_drv(
                _drv.cuMemExportToShareableHandle(handle, posix_fd, 0),
                "cuMemExportToShareableHandle(POSIX_FD)",
            )
            self.fds.append(int(fd))

        logger.info(
            "[vmm_fd] exported %d tensors over %d storages into %d arena(s), %.2f GiB resident",
            len(self.entries),
            len(storages),
            len(self.arena_sizes),
            self.resident_bytes / (1 << 30),
        )
        return self.entries

    def close(self) -> None:
        """Drop the daemon's copies. Clients that already imported keep theirs."""
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds = []
        self.views = {}
        for reservation in self._reservations:
            reservation.close(release_handles=True)
        self._reservations = []
        self._handles = []


class ArenaImporter:
    """Client side: import each arena fd, map it, and view tensors at their offsets."""

    def __init__(self, device_id: int):
        if not vmm_fd_available():
            raise RuntimeError(
                "the vmm_fd weight cache transport needs CUDA and the cuda-python "
                "driver bindings (cuda.bindings.driver)"
            )
        self.device_id = int(device_id)
        self._reservations: List[VmmReservation] = []
        self._handles: List[int] = []

    def import_arenas(self, fds: Sequence[int], sizes: Sequence[int]) -> None:
        if self._reservations:
            raise RuntimeError("ArenaImporter.import_arenas called twice")
        if len(fds) != len(sizes):
            raise RuntimeError(
                f"[vmm_fd] {len(fds)} descriptors for {len(sizes)} arenas"
            )
        torch.cuda.set_device(self.device_id)
        granularity = get_device_granularity(self.device_id)
        prop = make_device_allocation_prop(
            self.device_id, handle_types=int(_posix_fd_handle_type())
        )
        posix_fd = _posix_fd_handle_type()
        for fd, size in zip(fds, sizes):
            size = int(size)
            handle = check_drv(
                _drv.cuMemImportFromShareableHandle(int(fd), posix_fd),
                "cuMemImportFromShareableHandle(POSIX_FD)",
            )
            reservation = VmmReservation(
                size, prop, self.device_id, alignment=granularity
            )
            reservation.map_existing(0, size, handle)
            self._reservations.append(reservation)
            self._handles.append(int(handle))
            os.close(int(fd))

    def tensor(self, entry: Mapping[str, Any]) -> torch.Tensor:
        arena = int(entry["arena"])
        if not 0 <= arena < len(self._reservations):
            raise RuntimeError(
                f"[vmm_fd] entry names arena {arena} of {len(self._reservations)}"
            )
        return view_tensor(
            self._reservations[arena].base + int(entry["offset"]), entry, self.device_id
        )

    def close(self) -> None:
        """Unmap the arenas; the daemon's copies (and other clients') are unaffected."""
        for reservation in self._reservations:
            reservation.close(release_handles=False)
        for handle in self._handles:
            check_drv(_drv.cuMemRelease(handle), "cuMemRelease(imported arena)")
        self._reservations = []
        self._handles = []
