"""
GPU hardware detection.

Provides a value object describing the GPU and a factory function
to detect it via PyTorch CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GpuInfo:
    """Immutable value object describing detected GPU hardware."""

    name: str
    total_memory_bytes: int

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_bytes / (1024 ** 3)


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
