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
            # Engage a residency hold on each required model so a LATER admission
            # cannot evict it while this job is still using it (S1). The hold is
            # released by the lease. Under the admission lock the just-loaded,
            # protected models cannot be mid-eviction, so begin_operation
            # succeeds; roll back on the unexpected failure so no partial hold
            # leaks. Holds are taken BEFORE transient make_room so a transient
            # eviction pass can never pick a required model.
            held: list = []
            try:
                for key in dict.fromkeys(need.required_models):
                    model = self._registry[key].model
                    if not model.begin_operation():
                        raise RuntimeError(
                            f"could not hold required model '{key}' for the job"
                        )
                    held.append(model)
                if need.transient_bytes:
                    await self._make_room(need.transient_bytes, protect)
                    self._transient_reserved += need.transient_bytes
            except BaseException:
                for model in held:
                    model.end_operation()
                raise
            now = self._clock()
            for key in need.required_models:
                self._registry[key].last_used = now
        return GpuLease(self, need, held)

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

        # Reserve every *evictable* resident up front (loaded, unprotected, and
        # idle). try_reserve_for_eviction atomically excludes busy/held models,
        # so the policy only ever chooses among models we can actually free — a
        # busy LRU model can no longer shadow a freeable newer one. Reservations
        # not chosen as victims are cancelled below, leaving them untouched.
        reserved: list[tuple[str, _RegistryEntry]] = []
        for key, entry in self._registry.items():
            if key in protect or not entry.model.is_loaded():
                continue
            if entry.model.try_reserve_for_eviction():
                reserved.append((key, entry))

        candidates = [
            EvictionCandidate(
                key=key,
                estimated_vram_bytes=entry.model.estimated_vram_bytes,
                last_used=entry.last_used,
            )
            for key, entry in reserved
        ]
        victim_keys = set(self._policy.select_victims(candidates, shortfall))

        for key, entry in reserved:
            if key in victim_keys:
                logger.info(f"Evicting '{key}' to free VRAM for a pending workload")
                await entry.model.unload()
            else:
                entry.model.cancel_eviction_reservation()

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
    """Held for the duration of a job's GPU work. Releasing ends the residency
    hold on the required models (making them evictable again) and frees the
    transient reservation; the models stay LOADED (that is the point of
    residency). Release is synchronous and idempotent so it is cancellation-safe.

    MUST be used as an async context manager — ``async with await
    arbiter.acquire(...)`` — so release always runs. The ``__del__`` backstop
    only guards against a dropped lease leaking a hold/reservation; it is not a
    substitute for the context manager (GC timing is not deterministic)."""

    def __init__(self, arbiter: VramArbiter, need: WorkloadNeed, held_models: list):
        self._arbiter = arbiter
        self._need = need
        self._held_models = list(held_models)
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
        for model in self._held_models:
            model.end_operation()
        self._arbiter._release_transient(self._need.transient_bytes)

    def __del__(self):
        # Backstop only: a correctly-used lease is already released. Warn and
        # release so a dropped/never-entered lease cannot permanently leak a
        # residency hold or transient reservation (S3).
        if not self._released:
            try:
                logger.warning(
                    "GpuLease was garbage-collected without release(); use "
                    "'async with await arbiter.acquire(...)'. Releasing now."
                )
                self.release()
            except Exception:
                pass
