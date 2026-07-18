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
from .resident_model import ManagedModel, ResidentModel
from .vram_arbiter import GpuLease, VramArbiter, WorkloadNeed
from .vram_budget import VramBudget

__all__ = [
    "VramArbiter",
    "WorkloadNeed",
    "GpuLease",
    "ResidentModel",
    "ManagedModel",
    "EvictionPolicy",
    "LruEvictionPolicy",
    "EvictionCandidate",
    "VramBudget",
]
