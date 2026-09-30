"""
GPU round trip for the weight cache ``vmm_fd`` transport on one device.

A small module with tied weights, a transposed (non-contiguous) view and mixed
dtypes is exported through ArenaExporter (cuMemCreate arenas, cuMemcpyDtoD,
POSIX fd export), the daemon model is rebound onto the arena copies, then the
descriptors are imported through ArenaImporter in the same process the way an
engine would, and every tensor is compared bit for bit. The full daemon-plus-
engine path is exercised by test_weight_cache_daemon.py with
``--weight-cache-transport vmm_fd``.
"""

import os
import unittest

import torch

from sglang.srt.weight_cache import vmm_arena
from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
from sglang.srt.weight_cache.vmm_arena import (
    ArenaExporter,
    ArenaImporter,
    device_supports_posix_fd,
    vmm_fd_available,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(64, 32)
        self.proj = torch.nn.Linear(32, 48, bias=True)
        self.lm_head = torch.nn.Linear(32, 64, bias=False)
        self.lm_head.weight = self.embed.weight  # tied storage
        self.register_buffer("scale", torch.full((48,), 0.5, dtype=torch.float32))
        self.register_buffer(
            "cos_sin", torch.arange(24, dtype=torch.float16), persistent=False
        )


def _state_tensors(model):
    params = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    out = {}
    for name, tensor in model.state_dict().items():
        out[name] = (tensor.data, name in params)
    for name, buf in model.named_buffers():
        if name not in out:
            out[name] = (buf.data, False)
    return out


@unittest.skipUnless(vmm_fd_available(), "needs CUDA and cuda-python")
class TestVmmFdRoundTrip(CustomTestCase):
    def setUp(self):
        if not device_supports_posix_fd(0):
            self.skipTest("device 0 cannot export POSIX fd handles")
        torch.manual_seed(0)
        torch.cuda.set_device(0)
        self.model = Tiny().to("cuda:0").to(torch.bfloat16)
        # A non-contiguous parameter: a transposed view over its own storage.
        self.model.proj.weight = torch.nn.Parameter(
            self.model.proj.weight.data.t().contiguous().t(), requires_grad=False
        )
        self.model.scale = self.model.scale.float()
        self.originals = {
            n: t.detach().clone() for n, (t, _) in _state_tensors(self.model).items()
        }

    def test_export_rebind_import_matches_bit_for_bit(self):
        state = _state_tensors(self.model)
        exporter = ArenaExporter(0, arena_bytes=1 << 20)
        entries = exporter.export(state)
        self.addCleanup(exporter.close)

        self.assertEqual(set(entries), set(state))
        # Tied weights share one arena placement; the strided view keeps its strides.
        self.assertEqual(
            (entries["embed.weight"]["arena"], entries["embed.weight"]["offset"]),
            (entries["lm_head.weight"]["arena"], entries["lm_head.weight"]["offset"]),
        )
        self.assertEqual(
            entries["proj.weight"]["stride"], list(self.model.proj.weight.stride())
        )
        self.assertGreaterEqual(len(exporter.fds), 1)
        self.assertEqual(len(exporter.fds), len(exporter.arena_sizes))
        self.assertGreaterEqual(
            exporter.resident_bytes,
            sum(t.numel() * t.element_size() for t in self.originals.values()) // 2,
        )

        # Daemon side: rebind onto the arena copies and drop the originals.
        for name, (_, is_param) in state.items():
            IpcModelLoader._set_module_tensor(
                self.model, name, exporter.views[name], is_param=is_param
            )
        del state
        torch.cuda.synchronize()
        for name, (tensor, _) in _state_tensors(self.model).items():
            self.assertTrue(torch.equal(tensor, self.originals[name]), name)
        self.assertEqual(
            self.model.lm_head.weight.data_ptr(), self.model.embed.weight.data_ptr()
        )

        # Client side: the fds travel over SCM_RIGHTS in production; here dup them.
        fds = [os.dup(fd) for fd in exporter.fds]
        importer = ArenaImporter(0)
        importer.import_arenas(fds, exporter.arena_sizes)
        for name, entry in entries.items():
            tensor = importer.tensor(entry)
            self.assertEqual(tensor.shape, self.originals[name].shape, name)
            self.assertEqual(tensor.dtype, self.originals[name].dtype, name)
            self.assertEqual(tuple(tensor.stride()), tuple(entry["stride"]), name)
            self.assertTrue(torch.equal(tensor, self.originals[name]), name)
        imported_embed = importer.tensor(entries["embed.weight"])
        imported_head = importer.tensor(entries["lm_head.weight"])
        self.assertEqual(imported_embed.data_ptr(), imported_head.data_ptr())

        # The importer's mappings are independent of the exporter's: close the
        # exporter (the daemon going away) and the imported tensors still read.
        exporter.close()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(imported_embed, self.originals["embed.weight"]))
        self.assertTrue(
            torch.equal(importer.tensor(entries["scale"]), self.originals["scale"])
        )
        del imported_embed, imported_head
        importer.close()

    def test_plan_uses_the_device_granularity(self):
        granularity = vmm_arena.get_device_granularity(0)
        exporter = ArenaExporter(0, arena_bytes=1 << 20)
        exporter.export(_state_tensors(self.model))
        self.addCleanup(exporter.close)
        for size in exporter.arena_sizes:
            self.assertEqual(size % granularity, 0)


if __name__ == "__main__":
    unittest.main()
