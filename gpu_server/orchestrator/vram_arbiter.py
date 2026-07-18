"""
VRAM arbiter — the single admission point for GPU memory.

Replaces the server's implicit "one worker per modality" serialization with
explicit, capability-aware scheduling of a single physical VRAM pool. It owns
the oversubscription invariant and the lock that protects it; it *delegates* the
fit-arithmetic to ``VramBudget`` and the victim-choice to an ``EvictionPolicy``.
It arbitrates VRAM only — NVENC session limits are a different resource and stay
local to the video worker.

Every admission is serialized under one asyncio lock so the read-evict-reserve
sequence is atomic against other admissions. Because a need that forces eviction
of everything else is thereby exclusive on a small card but not on a large one,
serialization vs. concurrency is emergent from the budget — there is no
per-caller "exclusive" flag to get wrong.

Thread model: this runs on the event loop. It never touches CUDA directly — all
model load/unload is marshalled by ``ResidentModel`` onto each model's own
executor (see resident_model.py). The arbiter only ``await``s those.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..gpu_info import MemoryProbe
from ..logging_config import get_logger
from .eviction_policy import EvictionCandidate, EvictionPolicy, LruEvictionPolicy
from .resident_model import ManagedModel
from .vram_budget import VramBudget

logger = get_logger(__name__)


def _looks_like_oom(exc: BaseException) -> bool:
    """Whether an exception is a CUDA/allocator out-of-memory (torch or llama.cpp)."""
    name = type(exc).__name__
    if name in ("OutOfMemoryError", "CudaError"):
        return True
    return "out of memory" in str(exc).lower()


@dataclass(frozen=True)
class WorkloadNeed:
    """What a job needs from the GPU.

    ``required_models`` — keys that must be resident before the job runs.
    ``transient_bytes`` — ephemeral VRAM held only for the lease duration (e.g.
    NVENC for a video encode, or headroom for an LLM's growing KV cache), which
    is not a loaded model.
    """

    required_models: tuple[str, ...] = ()
    transient_bytes: int = 0


@dataclass
class _RegistryEntry:
    """Arbiter-owned bookkeeping for one resident. Scheduling state (last_used)
    lives HERE, not on the model — so the eviction policy can change without
    touching any model class."""

    model: ManagedModel
    last_used: float = 0.0


class VramArbiter:
    def __init__(
        self,
        probe: MemoryProbe,
        headroom_bytes: int,
        eviction_policy: Optional[EvictionPolicy] = None,
        clock: Callable[[], float] = time.monotonic,
        oom_predicate: Callable[[BaseException], bool] = _looks_like_oom,
    ):
        self._probe = probe
        self._headroom = headroom_bytes
        self._policy = eviction_policy or LruEvictionPolicy()
        self._clock = clock
        self._is_oom = oom_predicate
        self._registry: dict[str, _RegistryEntry] = {}
        self._transient_reserved = 0
        self._admission_lock = asyncio.Lock()

    # --- registration ------------------------------------------------------

    def register(self, model: ManagedModel) -> None:
        """Register a resident model. Call synchronously at startup, before any
        worker can acquire (F9). Recreation after a stuck-timeout should rebind
        the SAME model object (keeping identity) rather than re-register."""
        self._registry[model.key] = _RegistryEntry(model=model)

    # --- admission ---------------------------------------------------------

    async def acquire(self, need: WorkloadNeed) -> "GpuLease":
        """Admit a job: ensure its required models are resident (evicting idle
        ones to make room), reserve any transient headroom, and return a lease.
        The returned lease MUST be used as an async context manager so its
        transient reservation is released. Serialized against other admissions.

        Callers should invoke this OUTSIDE any per-request processing timeout
        (F4): time spent waiting here for another job's VRAM is not the job's
        own GPU time.
        """
        unknown = [k for k in need.required_models if k not in self._registry]
        if unknown:
            raise KeyError(f"Unknown required model(s): {unknown}")

        protect = set(need.required_models)
        async with self._admission_lock:
            for key in need.required_models:
                await self._ensure_loaded(key, protect)
            if need.transient_bytes:
                await self._make_room(need.transient_bytes, protect)
                self._transient_reserved += need.transient_bytes
            now = self._clock()
            for key in need.required_models:
                self._registry[key].last_used = now
        return GpuLease(self, need)

    async def _ensure_loaded(self, key: str, protect: set[str]) -> None:
        entry = self._registry[key]
        if entry.model.is_loaded():
            return
        need_bytes = entry.model.estimated_vram_bytes
        await self._make_room(need_bytes, protect)
        try:
            await entry.model.load()
        except BaseException as exc:  # noqa: BLE001 - re-raised unless OOM
            if not self._is_oom(exc):
                raise
            # Out of memory despite our accounting — free everything evictable
            # and try once more before giving up (F6).
            logger.warning(
                f"Load of '{key}' hit OOM; evicting aggressively and retrying"
            )
            await self._make_room(need_bytes, protect, aggressive=True)
            await entry.model.load()

    async def _make_room(
        self, need_bytes: int, protect: set[str], aggressive: bool = False
    ) -> None:
        budget = self._current_budget()
        if not aggressive and budget.can_fit(need_bytes):
            return
        shortfall = budget.total_bytes if aggressive else budget.shortfall_bytes(need_bytes)

        candidates = [
            EvictionCandidate(
                key=k,
                estimated_vram_bytes=e.model.estimated_vram_bytes,
                last_used=e.last_used,
            )
            for k, e in self._registry.items()
            if k not in protect and e.model.is_loaded()
        ]
        victim_keys = self._policy.select_victims(candidates, shortfall)

        for vkey in victim_keys:
            model = self._registry[vkey].model
            # Skip a model that is busy or already being evicted; we can only
            # unload one we can atomically reserve as idle.
            if model.try_reserve_for_eviction():
                logger.info(f"Evicting '{vkey}' to free VRAM for a pending workload")
                await model.unload()

    def _current_budget(self) -> VramBudget:
        snap = self._probe.snapshot()
        reserved_resident = sum(
            e.model.estimated_vram_bytes
            for e in self._registry.values()
            if e.model.is_loaded()
        )
        return VramBudget(
            total_bytes=snap.total_bytes,
            free_bytes=snap.free_bytes,
            reserved_resident_bytes=reserved_resident,
            reserved_transient_bytes=self._transient_reserved,
            headroom_bytes=self._headroom,
        )

    # --- lease release -----------------------------------------------------

    def _release_transient(self, transient_bytes: int) -> None:
        """Synchronous, non-awaiting release so a cancellation cannot interrupt
        it and leak a reservation (F4). Clamped at zero defensively."""
        if transient_bytes:
            self._transient_reserved = max(0, self._transient_reserved - transient_bytes)

    # --- introspection (tests / diagnostics) -------------------------------

    @property
    def transient_reserved_bytes(self) -> int:
        return self._transient_reserved

    def loaded_keys(self) -> list[str]:
        return [k for k, e in self._registry.items() if e.model.is_loaded()]


class GpuLease:
    """Held for the duration of a job's GPU work. Releasing frees the transient
    reservation; resident models stay loaded (that is the point of residency).
    Release is synchronous and idempotent so it is cancellation-safe."""

    def __init__(self, arbiter: VramArbiter, need: WorkloadNeed):
        self._arbiter = arbiter
        self._need = need
        self._released = False

    async def __aenter__(self) -> "GpuLease":
        return self

    async def __aexit__(self, *exc_info) -> bool:
        self.release()
        return False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._arbiter._release_transient(self._need.transient_bytes)
