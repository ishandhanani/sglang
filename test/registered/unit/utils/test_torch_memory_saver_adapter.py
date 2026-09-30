"""
CPU-only unit tests for the torch_memory_saver hook-mode seam
(``--memory-saver-hook-mode``, RFC #27310 section 4.1).

They drive ``TorchMemorySaverAdapter`` against a scripted stand-in for the
torch_memory_saver singleton, so they need no CUDA and no real allocator:

  - mode resolution: explicit argument > server argument > platform default
    (torch on XPU, preload elsewhere)
  - preload mode LD_PRELOADs scheduler subprocesses and hands CUDA-graph
    capture to torch_memory_saver; torch mode does neither (one warning)
  - one hook mode per process: a second adapter asking for a different mode
    is refused, a singleton initialized before SGLang configured it is reported
  - the server-argument validation that ties the flag to --enable-memory-saver
    and keeps preload off Intel XPU
"""

import logging
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.validation_hook import validate_memory_saver_hook_mode
from sglang.srt.utils import torch_memory_saver_adapter as adapter_mod
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeSingleton:
    """torch_memory_saver.torch_memory_saver: takes a hook mode once, then refuses."""

    def __init__(self, initialized: bool = False):
        self._ctor_kwargs = None if initialized else {}
        self.calls = []

    @property
    def hook_mode(self):
        raise AttributeError

    @hook_mode.setter
    def hook_mode(self, mode):
        assert self._ctor_kwargs is not None, "Cannot configure after initialization"
        self._ctor_kwargs["hook_mode"] = mode

    @property
    def configured_mode(self):
        return (self._ctor_kwargs or {}).get("hook_mode")

    @contextmanager
    def region(self, tag, enable_cpu_backup=False):
        self.calls.append(("region", tag, enable_cpu_backup))
        yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        self.calls.append(("cuda_graph", kwargs.get("tag")))
        yield

    def pause(self, tag):
        self.calls.append(("pause", tag))

    def resume(self, tag):
        self.calls.append(("resume", tag))

    def disable(self):
        self.calls.append(("disable",))

    @property
    def enabled(self):
        return True


class FakeModule:
    """The torch_memory_saver module: configure_subprocess is the LD_PRELOAD hook."""

    def __init__(self, singleton):
        self.torch_memory_saver = singleton
        self.preload_calls = 0

    @contextmanager
    def configure_subprocess(self):
        self.preload_calls += 1
        yield


class TestTorchMemorySaverAdapterHookMode(CustomTestCase):
    def setUp(self):
        self.singleton = FakeSingleton()
        self.module = FakeModule(self.singleton)
        self._patches = [
            patch.object(adapter_mod, "torch_memory_saver", self.module),
            patch.object(adapter_mod, "_memory_saver", self.singleton),
            patch.object(adapter_mod, "import_error", None),
            patch.object(adapter_mod, "_applied_hook_mode", None),
            patch.object(adapter_mod, "_warned_unpauseable_cuda_graph", False),
            patch.object(adapter_mod, "is_xpu", lambda: False),
            patch.object(adapter_mod, "configured_hook_mode", lambda: None),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    # -- resolution -----------------------------------------------------------

    def test_default_is_preload_and_preloads_subprocesses(self):
        adapter = TorchMemorySaverAdapter.create(enable=True)
        self.assertEqual(adapter.hook_mode, "preload")
        self.assertEqual(self.singleton.configured_mode, "preload")
        self.assertEqual(adapter_mod.applied_hook_mode(), "preload")
        with adapter.configure_subprocess():
            pass
        self.assertEqual(self.module.preload_calls, 1)
        with adapter.cuda_graph(tag="cuda_graph"):
            pass
        self.assertIn(("cuda_graph", "cuda_graph"), self.singleton.calls)

    def test_xpu_defaults_to_torch(self):
        with patch.object(adapter_mod, "is_xpu", lambda: True):
            adapter = TorchMemorySaverAdapter.create(enable=True)
        self.assertEqual(adapter.hook_mode, "torch")
        self.assertEqual(self.singleton.configured_mode, "torch")

    def test_server_argument_wins_over_platform_default(self):
        with patch.object(adapter_mod, "configured_hook_mode", lambda: "torch"):
            adapter = TorchMemorySaverAdapter.create(enable=True)
        self.assertEqual(adapter.hook_mode, "torch")

    def test_explicit_argument_wins_over_server_argument(self):
        with patch.object(adapter_mod, "configured_hook_mode", lambda: "torch"):
            adapter = TorchMemorySaverAdapter.create(enable=True, hook_mode="preload")
        self.assertEqual(adapter.hook_mode, "preload")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            TorchMemorySaverAdapter.create(enable=True, hook_mode="dlopen")

    # -- torch mode behavior --------------------------------------------------

    def test_torch_mode_skips_ld_preload_and_graph_capture(self):
        adapter = TorchMemorySaverAdapter.create(enable=True, hook_mode="torch")
        with adapter.configure_subprocess():
            pass
        self.assertEqual(self.module.preload_calls, 0)
        with self.assertLogs(adapter_mod.logger, level=logging.WARNING) as logs:
            with adapter.cuda_graph(tag="cuda_graph"):
                pass
            with adapter.cuda_graph(tag="cuda_graph"):
                pass
        self.assertEqual(len(logs.records), 1)  # warned once, not per capture
        self.assertNotIn(("cuda_graph", "cuda_graph"), self.singleton.calls)
        # region / pause / resume are the same in both modes.
        with adapter.region("weights"):
            pass
        adapter.pause("weights")
        adapter.resume("weights")
        self.assertEqual(
            self.singleton.calls,
            [("region", "weights", False), ("pause", "weights"), ("resume", "weights")],
        )

    # -- one mode per process -------------------------------------------------

    def test_later_adapters_without_a_request_follow_the_applied_mode(self):
        TorchMemorySaverAdapter.create(enable=True, hook_mode="torch")
        again = TorchMemorySaverAdapter.create(
            enable=True
        )  # would default to preload on its own
        self.assertEqual(again.hook_mode, "torch")
        self.assertEqual(self.singleton.configured_mode, "torch")

    def test_second_adapter_with_a_different_mode_is_refused(self):
        TorchMemorySaverAdapter.create(enable=True, hook_mode="preload")
        with self.assertRaisesRegex(
            RuntimeError, "already configured with hook mode 'preload'"
        ):
            TorchMemorySaverAdapter.create(enable=True, hook_mode="torch")

    def test_same_mode_twice_does_not_touch_the_singleton_again(self):
        TorchMemorySaverAdapter.create(enable=True, hook_mode="preload")
        self.singleton._ctor_kwargs = None  # TMS initialized in between
        adapter = TorchMemorySaverAdapter.create(enable=True, hook_mode="preload")
        self.assertEqual(adapter.hook_mode, "preload")

    def test_singleton_initialized_before_sglang_is_reported(self):
        early = FakeSingleton(initialized=True)
        with patch.object(adapter_mod, "_memory_saver", early):
            with self.assertRaisesRegex(
                RuntimeError, "initialized before SGLang configured its hook mode"
            ):
                TorchMemorySaverAdapter.create(enable=True)

    # -- disabled -------------------------------------------------------------

    def test_noop_adapter_has_no_mode_and_configures_nothing(self):
        adapter = TorchMemorySaverAdapter.create(enable=False)
        self.assertIsNone(adapter.hook_mode)
        self.assertFalse(adapter.enabled)
        with adapter.configure_subprocess():
            pass
        self.assertIsNone(adapter_mod.applied_hook_mode())
        self.assertIsNone(self.singleton.configured_mode)


class TestMemorySaverHookModeValidation(CustomTestCase):
    @staticmethod
    def _cfg(**overrides):
        base = dict(
            enable_memory_saver=True, memory_saver_hook_mode=None, device="cuda"
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_unset_mode_is_always_fine(self):
        validate_memory_saver_hook_mode(self._cfg(enable_memory_saver=False))
        validate_memory_saver_hook_mode(self._cfg(device="xpu"))

    def test_mode_without_memory_saver_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires --enable-memory-saver"):
            validate_memory_saver_hook_mode(
                self._cfg(enable_memory_saver=False, memory_saver_hook_mode="torch")
            )

    def test_preload_on_xpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not available on Intel XPU"):
            validate_memory_saver_hook_mode(
                self._cfg(memory_saver_hook_mode="preload", device="xpu")
            )

    def test_torch_on_xpu_and_either_mode_on_cuda_are_accepted(self):
        validate_memory_saver_hook_mode(
            self._cfg(memory_saver_hook_mode="torch", device="xpu")
        )
        validate_memory_saver_hook_mode(self._cfg(memory_saver_hook_mode="torch"))
        validate_memory_saver_hook_mode(self._cfg(memory_saver_hook_mode="preload"))


if __name__ == "__main__":
    unittest.main()
