"""
GPU orchestration: capability-aware arbitration of a single VRAM pool across
resident models (whisper, pyannote, LLM) and transient workloads (video/NVENC).

Public surface:
- ``VramArbiter`` / ``WorkloadNeed`` / ``GpuLease`` — admission control.
- ``ResidentModel`` / ``ManagedModel`` — per-model VRAM residency.
- ``EvictionPolicy`` / ``LruEvictionPolicy`` — victim choice (Strategy).
- ``VramBudget`` — pure fit arithmetic.
"""
from .eviction_policy import EvictionCandidate, EvictionPolicy, LruEvictionPolicy
from .resident_model import ManagedModel, ResidentBinding, ResidentModel
from .resident_processor_handle import ResidentCapable, ResidentProcessorHandle
from .vram_arbiter import GpuLease, VramArbiter, WorkloadNeed
from .vram_budget import VramBudget

__all__ = [
    "VramArbiter",
    "WorkloadNeed",
    "GpuLease",
    "ResidentModel",
    "ManagedModel",
    "ResidentBinding",
    "ResidentCapable",
    "ResidentProcessorHandle",
    "EvictionPolicy",
    "LruEvictionPolicy",
    "EvictionCandidate",
    "VramBudget",
]
