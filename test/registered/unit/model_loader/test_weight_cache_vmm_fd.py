"""
CPU-only unit tests for the weight cache ``vmm_fd`` transport
(``--weight-cache-transport``, RFC #27310 section 4.2 phase 1).

They cover the parts that need no GPU:

  - arena planning: first-fit placement, storage alignment, granularity
    rounding, oversized storages, empty storages, argument validation
  - the wire: the response carries the arena sizes, the descriptors follow
    over SCM_RIGHTS in index order and reach the client as live fds, and the
    client refuses malformed size lists and duplicate indices
  - transport selection for the daemon: ``auto`` and ``torch_ipc`` keep the
    legacy backend, ``vmm_fd`` is refused (not silently downgraded) where it
    cannot work, unknown names are rejected

The GPU round trip (real cuMemCreate, export, import, tensor views, rebinding)
lives in test_weight_cache_vmm_fd_gpu.py.
"""

import os
import socket
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.weight_cache import transport as transport_mod
from sglang.srt.weight_cache import vmm_arena
from sglang.srt.weight_cache.protocol import recv_msg
from sglang.srt.weight_cache.transport import (
    AUTO_TRANSPORT,
    TORCH_IPC_BACKEND,
    VMM_FD_BACKEND,
    TorchIpcTransportBackend,
    VmmFdTransportBackend,
    choose_daemon_transport_backend,
    get_client_transport_backend,
)
from sglang.srt.weight_cache.vmm_arena import plan_arenas
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

GRANULARITY = 1024


class TestPlanArenas(CustomTestCase):
    def test_first_fit_with_aligned_offsets(self):
        sizes, placements = plan_arenas(
            [100, 200], arena_bytes=4000, granularity=GRANULARITY, alignment=512
        )
        self.assertEqual(placements, [(0, 0), (0, 512)])
        self.assertEqual(sizes, [1024])  # cursor 712, rounded to the granularity

    def test_opens_a_new_arena_when_the_open_one_is_full(self):
        sizes, placements = plan_arenas(
            [900, 200, 50], arena_bytes=1000, granularity=GRANULARITY, alignment=512
        )
        # 900 fills arena 0 (next aligned start 1024 > 1000); 200 opens arena 1;
        # 50 goes back into arena 1 at 512 (first fit), not into a third arena.
        self.assertEqual(placements, [(0, 0), (1, 0), (1, 512)])
        self.assertEqual(sizes, [1024, 1024])

    def test_oversized_storage_gets_its_own_arena(self):
        sizes, placements = plan_arenas(
            [3000, 10], arena_bytes=1000, granularity=GRANULARITY, alignment=512
        )
        self.assertEqual(placements, [(0, 0), (1, 0)])
        self.assertEqual(sizes, [3072, 1024])

    def test_empty_storage_still_has_a_place(self):
        sizes, placements = plan_arenas([0], arena_bytes=1000, granularity=GRANULARITY)
        self.assertEqual(placements, [(0, 0)])
        self.assertEqual(sizes, [1024])

    def test_no_storages_means_no_arenas(self):
        self.assertEqual(
            plan_arenas([], arena_bytes=1000, granularity=GRANULARITY), ([], [])
        )

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            plan_arenas([1], arena_bytes=0, granularity=GRANULARITY)
        with self.assertRaises(ValueError):
            plan_arenas([1], arena_bytes=10, granularity=0)
        with self.assertRaises(ValueError):
            plan_arenas([-1], arena_bytes=10, granularity=GRANULARITY)


class FakeImporter:
    """Stands in for ArenaImporter: records what the client would map."""

    instances = []

    def __init__(self, device_id):
        self.device_id = device_id
        self.fds = None
        self.sizes = None
        self.tensors = []
        FakeImporter.instances.append(self)

    def import_arenas(self, fds, sizes):
        self.fds = list(fds)
        self.sizes = list(sizes)

    def tensor(self, entry):
        self.tensors.append(entry)
        return ("tensor", entry["arena"], entry["offset"])

    def close(self):
        pass


class TestVmmFdWire(CustomTestCase):
    def setUp(self):
        FakeImporter.instances = []
        self._importer_patch = patch.object(
            transport_mod, "ArenaImporter", FakeImporter
        )
        self._importer_patch.start()
        self.addCleanup(self._importer_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.files = []
        self.fds = []
        for i in range(2):
            path = os.path.join(self.tmp.name, f"arena{i}")
            with open(path, "w") as f:
                f.write(f"arena {i}")
            fd = os.open(path, os.O_RDONLY)
            self.files.append(path)
            self.fds.append(fd)
            self.addCleanup(os.close, fd)

    def _daemon_backend(self):
        backend = VmmFdTransportBackend(device_id=0)
        backend._exporter = SimpleNamespace(
            fds=list(self.fds), arena_sizes=[4096, 8192], views={}, resident_bytes=12288
        )
        return backend

    def test_response_and_descriptors_round_trip(self):
        daemon = self._daemon_backend()
        entries = {
            "w": {
                "arena": 1,
                "offset": 512,
                "storage_bytes": 64,
                "shape": [4, 8],
                "stride": [8, 1],
                "storage_offset": 0,
                "dtype": "bfloat16",
                "is_param": True,
            },
        }
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        daemon.send_fetch_state_response(
            a,
            config={"tp_size": 1},
            entries=entries,
            pid=4242,
            preloaded_weights_bytes=7,
        )

        result = recv_msg(b)
        self.assertEqual(result["transport_backend"], VMM_FD_BACKEND)
        self.assertEqual(result["arenas"], [4096, 8192])
        self.assertEqual(result["preloaded_weights_bytes"], 7)
        client = get_client_transport_backend(result["transport_backend"])
        self.assertIsInstance(client, VmmFdTransportBackend)
        client.bind_device(3)
        client.recv_fetch_state_response(b, result)

        (importer,) = FakeImporter.instances
        self.assertEqual(importer.device_id, 3)
        self.assertEqual(importer.sizes, [4096, 8192])
        # The received descriptors are new fds that point at the same files, in index order.
        self.assertEqual(len(importer.fds), 2)
        for received, original in zip(importer.fds, self.fds):
            self.assertNotEqual(received, original)
            self.assertEqual(os.fstat(received).st_ino, os.fstat(original).st_ino)
            os.close(received)
        self.assertEqual(client.import_tensor(entries["w"]), ("tensor", 1, 512))

    def test_client_refuses_malformed_arena_sizes(self):
        client = VmmFdTransportBackend(device_id=0)
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        for bad in (
            {},
            {"arenas": "x"},
            {"arenas": [0]},
            {"arenas": [True]},
            {"arenas": [-1]},
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid arena sizes"):
                client.recv_fetch_state_response(b, bad)
        self.assertEqual(FakeImporter.instances, [])

    def test_client_refuses_a_duplicate_descriptor_index(self):
        client = VmmFdTransportBackend(device_id=0)
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        transport_mod._send_fd(a, self.fds[0], 0)
        transport_mod._send_fd(a, self.fds[1], 0)  # index 0 again
        with self.assertRaisesRegex(RuntimeError, "unexpected arena descriptor index"):
            client.recv_fetch_state_response(b, {"arenas": [4096, 8192]})
        self.assertEqual(FakeImporter.instances, [])

    def test_send_before_export_is_an_error(self):
        a, b = socket.socketpair()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        with self.assertRaisesRegex(RuntimeError, "before prepare_export"):
            VmmFdTransportBackend().send_fetch_state_response(
                a, config={}, entries={}, pid=1
            )
        with self.assertRaisesRegex(RuntimeError, "before recv_fetch_state_response"):
            VmmFdTransportBackend().import_tensor({})


class TestDaemonTransportSelection(CustomTestCase):
    def test_auto_and_torch_ipc_keep_the_legacy_backend(self):
        for requested in (AUTO_TRANSPORT, TORCH_IPC_BACKEND):
            backend = choose_daemon_transport_backend({}, requested)
            self.assertIsInstance(backend, TorchIpcTransportBackend)
            self.assertEqual(backend.name, TORCH_IPC_BACKEND)

    def test_vmm_fd_is_refused_where_it_cannot_work(self):
        with patch.object(
            VmmFdTransportBackend, "can_export_state", classmethod(lambda cls, s: False)
        ):
            with self.assertRaisesRegex(RuntimeError, "vmm_fd needs"):
                choose_daemon_transport_backend({}, VMM_FD_BACKEND)

    def test_vmm_fd_is_selected_when_it_can_work(self):
        with patch.object(
            VmmFdTransportBackend, "can_export_state", classmethod(lambda cls, s: True)
        ):
            backend = choose_daemon_transport_backend({}, VMM_FD_BACKEND, device_id=1)
        self.assertIsInstance(backend, VmmFdTransportBackend)
        self.assertEqual(backend._device(), 1)

    def test_unknown_transport_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_daemon_transport_backend({}, "carrier-pigeon")

    def test_can_export_state_is_false_without_cuda_tensors(self):
        with patch.object(vmm_arena, "vmm_fd_available", lambda: True):
            with patch.object(transport_mod, "vmm_fd_available", lambda: True):
                self.assertFalse(VmmFdTransportBackend.can_export_state({}))


if __name__ == "__main__":
    unittest.main()
