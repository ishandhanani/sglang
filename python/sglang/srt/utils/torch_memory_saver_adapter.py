"""SGLang's seam onto ``torch_memory_saver``: one hook mode per process, tag-based pause/resume.

``torch_memory_saver`` (TMS) can intercept allocations in two ways. ``preload`` swaps
``cudaMalloc`` through ``LD_PRELOAD`` in every scheduler subprocess and is the only
mode that can pause CUDA-graph memory. ``torch`` registers an in-process pluggable
allocator: no ``LD_PRELOAD``, graph memory stays resident, and an external memory
owner (a weight cache or a VMM memory service) can sit behind the same allocator
callbacks. The mode is chosen once per process (``--memory-saver-hook-mode``, else
the platform default) before TMS's singleton initializes on first use; every
adapter created afterwards must agree.
"""

import logging
from abc import ABC
from contextlib import contextmanager
from typing import Optional

from sglang.srt.utils.common import is_xpu

try:
    import torch_memory_saver

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    torch_memory_saver = None
    _memory_saver = None
    import_error = e

logger = logging.getLogger(__name__)

HOOK_MODE_PRELOAD = "preload"
HOOK_MODE_TORCH = "torch"
HOOK_MODES = (HOOK_MODE_PRELOAD, HOOK_MODE_TORCH)

# The hook mode this process configured TMS's singleton with. TMS takes the mode
# once, before its first region/pause/resume; later adapters are checked against it.
_applied_hook_mode: Optional[str] = None
_warned_unpauseable_cuda_graph = False


def default_hook_mode() -> str:
    """The mode used when the server does not choose one.

    Intel XPU only ships the in-process pluggable allocator (the LD_PRELOAD path is
    CUDA/HIP-only); everywhere else ``preload`` keeps CUDA-graph memory pauseable.
    """
    return HOOK_MODE_TORCH if is_xpu() else HOOK_MODE_PRELOAD


def configured_hook_mode() -> Optional[str]:
    """``--memory-saver-hook-mode`` from the published exec config; None when unset or not published."""
    try:
        from sglang.srt.runtime_context import get_exec

        return get_exec().features.memory_saver_hook_mode
    except (ImportError, ValueError, AttributeError):
        # Not published yet (a bare adapter in a test or a tool), or an older
        # config without the field: fall back to the platform default.
        return None


def resolve_hook_mode(hook_mode: Optional[str] = None) -> str:
    """Explicit request, else the server argument, else the platform default."""
    mode = hook_mode or configured_hook_mode() or default_hook_mode()
    if mode not in HOOK_MODES:
        raise ValueError(
            f"unknown torch_memory_saver hook mode {mode!r}; expected one of {HOOK_MODES}"
        )
    return mode


def applied_hook_mode() -> Optional[str]:
    """The mode TMS was configured with in this process, or None before the first enabled adapter."""
    return _applied_hook_mode


def _apply_hook_mode(mode: str) -> None:
    global _applied_hook_mode
    if _applied_hook_mode is not None:
        if _applied_hook_mode != mode:
            raise RuntimeError(
                f"torch_memory_saver is already configured with hook mode {_applied_hook_mode!r} in this "
                f"process; it cannot switch to {mode!r}. Every memory saver adapter of a process shares "
                "one TMS singleton, so choose the mode once with --memory-saver-hook-mode."
            )
        return
    try:
        _memory_saver.hook_mode = mode
    except AssertionError as e:
        # TMS refuses to be configured after its singleton initialized. That means
        # something used torch_memory_saver directly before SGLang's first adapter.
        raise RuntimeError(
            "torch_memory_saver was initialized before SGLang configured its hook mode; "
            "create the memory saver adapter (or set the hook mode) before any region(), "
            "pause() or resume() call in this process"
        ) from e
    _applied_hook_mode = mode
    logger.info(
        "torch_memory_saver hook mode: %s (%s)",
        mode,
        "LD_PRELOAD in scheduler subprocesses"
        if mode == HOOK_MODE_PRELOAD
        else "in-process pluggable allocator, no LD_PRELOAD, CUDA-graph memory not pauseable",
    )


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool, hook_mode: Optional[str] = None):
        """The process's memory saver, or a no-op when the memory saver is off.

        ``hook_mode`` overrides ``--memory-saver-hook-mode``; leave it None to follow
        the server argument (and the platform default below it). The first enabled
        adapter of a process fixes the mode for every later one.
        """
        if not enable:
            return _TorchMemorySaverAdapterNoop()
        if import_error is not None:
            if is_xpu():
                # XPU ships no prebuilt wheel; it is built from source against the
                # local oneAPI + torch-XPU runtime. TMS_PLATFORM=xpu forces the XPU
                # backend; --no-build-isolation lets the build see torch and match
                # the libsycl ABI to it.
                logger.warning(
                    "enable_memory_saver is enabled, but torch-memory-saver is "
                    "not installed. On Intel XPU, build it from source with Intel "
                    "oneAPI on PATH: `TMS_PLATFORM=xpu pip3 install "
                    "--no-build-isolation git+https://github.com/fzyzcjy/"
                    "torch_memory_saver.git@a5c99f11b18ebb8e9fda71a68812e476ae49e417`."
                )
            else:
                logger.warning(
                    "enable_memory_saver is enabled, but "
                    "torch-memory-saver is not installed. Please install it "
                    "via `pip3 install torch-memory-saver`. "
                )
            raise import_error
        # An adapter with no request of its own follows the mode this process
        # already runs; only an explicit, different request is refused.
        mode = hook_mode or _applied_hook_mode or resolve_hook_mode(None)
        if mode not in HOOK_MODES:
            raise ValueError(
                f"unknown torch_memory_saver hook mode {mode!r}; expected one of {HOOK_MODES}"
            )
        _apply_hook_mode(mode)
        return _TorchMemorySaverAdapterReal(mode)

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    @property
    def hook_mode(self) -> Optional[str]:
        """``preload`` or ``torch`` for a live adapter, None for the no-op one."""
        raise NotImplementedError

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self, tag: str, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def cuda_graph(self, **kwargs):
        raise NotImplementedError

    def disable(self):
        raise NotImplementedError

    def pause(self, tag: str):
        raise NotImplementedError

    def resume(self, tag: str):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control.

    Backed by the upstream torch_memory_saver package (CUDA VMM, and Intel XPU via
    Level Zero). In ``torch`` hook mode (always on XPU, opt-in elsewhere) there is
    nothing to LD_PRELOAD and CUDA-graph memory cannot be paused, so
    configure_subprocess() and cuda_graph() are no-ops there; region/pause/resume
    are fully supported in both modes.
    """

    def __init__(self, hook_mode: str):
        self._hook_mode = hook_mode

    @property
    def hook_mode(self) -> Optional[str]:
        return self._hook_mode

    def configure_subprocess(self):
        if self._hook_mode != HOOK_MODE_PRELOAD:
            # Nothing to preload: the pluggable allocator registers in-process.
            return self._noop_context()
        return torch_memory_saver.configure_subprocess()

    def region(self, tag: str, enable_cpu_backup: bool = False):
        return _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)

    def cuda_graph(self, **kwargs):
        if self._hook_mode != HOOK_MODE_PRELOAD:
            # Upstream gates pauseable graph capture on hook_mode="preload". Warn
            # rather than raise, so a graph backend that routes here surfaces that
            # graph memory is not pauseable instead of failing to launch.
            global _warned_unpauseable_cuda_graph
            if not _warned_unpauseable_cuda_graph:
                _warned_unpauseable_cuda_graph = True
                logger.warning(
                    "torch_memory_saver hook mode %r cannot make CUDA-graph memory pauseable; "
                    "graph allocations will not be released by "
                    "release_memory_occupation(tags=['cuda_graph']).",
                    self._hook_mode,
                )
            return self._noop_context()
        return _memory_saver.cuda_graph(**kwargs)

    @contextmanager
    def _noop_context(self, **kwargs):
        yield

    def disable(self):
        return _memory_saver.disable()

    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        return _memory_saver.resume(tag=tag)

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @property
    def hook_mode(self) -> Optional[str]:
        return None

    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        yield

    @contextmanager
    def disable(self):
        yield

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
