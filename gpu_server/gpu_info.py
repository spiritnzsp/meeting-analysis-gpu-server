"""
GPU hardware detection.

Provides value objects describing the GPU (static facts) and its live memory
state, plus a probe abstraction the orchestrator depends on for admission
decisions. The static/live split is deliberate: ``GpuInfo`` is a one-shot
description that never changes; ``GpuSnapshot`` is a moment-in-time reading that
the arbiter re-takes on every admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GpuInfo:
    """Immutable value object describing detected GPU hardware (static facts)."""

    name: str
    total_memory_bytes: int

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_bytes / (1024 ** 3)


@dataclass(frozen=True)
class GpuSnapshot:
    """A moment-in-time reading of GPU memory (driver-level free/total).

    ``free_bytes`` comes from the driver and reflects *everything* using the
    card — our loaded models, other processes (e.g. the desktop app), and NVENC
    — but it is only a lower bound: torch's caching allocator holds freed blocks
    as reserved, and the value can drift between the read and an actual load.
    Treat it as one input to admission, never as ground truth.
    """

    total_bytes: int
    free_bytes: int

    @property
    def free_gb(self) -> float:
        return self.free_bytes / (1024 ** 3)


class MemoryProbe(Protocol):
    """Reads live GPU memory. An abstraction so the arbiter can be unit-tested
    with a scripted fake instead of a real CUDA device."""

    def snapshot(self) -> GpuSnapshot:
        ...


class TorchMemoryProbe:
    """Live GPU memory probe backed by ``torch.cuda.mem_get_info``.

    NOTE: the first call may initialise the CUDA context (blocking); callers on
    the event loop should run the first probe in an executor (see F9).
    """

    def __init__(self, device: int = 0):
        self._device = device

    def snapshot(self) -> GpuSnapshot:
        import torch
        free, total = torch.cuda.mem_get_info(self._device)
        return GpuSnapshot(total_bytes=int(total), free_bytes=int(free))


def detect_gpu() -> Optional[GpuInfo]:
    """Detect GPU via torch.cuda.

    Returns:
        GpuInfo if a CUDA GPU is available, None otherwise.
    """
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return GpuInfo(
                name=torch.cuda.get_device_name(0),
                total_memory_bytes=props.total_memory,
            )
        logger.warning("CUDA not available - will use CPU (slow!)")
        return None
    except ImportError:
        logger.warning("PyTorch not installed - GPU detection skipped")
        return None
