"""
Resident model — ownership of one GPU model's VRAM residency lifecycle.

This is the polymorphism the server lacked: a uniform contract the arbiter can
load, unload, and reason about, independent of whether the underlying model is
whisper, a pyannote pipeline, an embedding model, or an LLM. Residency is
deliberately separate from *processing* (BaseProcessor's executor/cancel/idle):
a pyannote processor holds TWO evictable models, video holds none.

Thread-safety is the whole game here (see the concurrency review):

* F1/F2 — the actual model teardown (``del model; empty_cache()``) is a torch
  call that MUST NOT run on the event loop while inference for that model may be
  in flight. ``load``/``unload`` are async wrappers that marshal the blocking
  work onto the model's OWN single-thread executor, so it serialises behind any
  in-progress inference on that same queue. The arbiter only ``await``s.
* F3/F8 — a ``threading.Lock`` (shared with the executor thread, so NOT an
  asyncio primitive) guards ``_loaded``/``_busy``/``_evicting``. Eviction is
  reserve-then-act: ``try_reserve_for_eviction`` atomically confirms the model
  is loaded, idle, and not already being evicted before the arbiter unloads it,
  and ``begin_operation`` refuses to start work on a model reserved for
  eviction. This closes the check-then-act race between "is it idle?" and the
  unload.
* F5 — the key is stable and the inner model reference is mutable
  (``rebind``), so a processor recreated after a stuck-timeout keeps the same
  registry identity; only the loaded flag resets.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Executor
from typing import Callable, Protocol

from ..logging_config import get_logger

logger = get_logger(__name__)


class ManagedModel(Protocol):
    """Residency contract the arbiter depends on. Intentionally slim: only what
    it takes to load, unload, and size a model. Scheduling state (last_used) and
    the choice to evict live in the arbiter, not here."""

    key: str
    estimated_vram_bytes: int

    def is_loaded(self) -> bool: ...
    async def load(self) -> None: ...
    async def unload(self) -> None: ...
    def try_reserve_for_eviction(self) -> bool: ...
    def cancel_eviction_reservation(self) -> None: ...
    def begin_operation(self) -> bool: ...
    def end_operation(self) -> None: ...


class ResidentModel:
    """Concrete ``ManagedModel``. ``load_fn``/``unload_fn`` are blocking callables
    that touch the GPU and are always executed on ``executor`` (the owning
    processor's single-thread pool)."""

    def __init__(
        self,
        key: str,
        estimated_vram_bytes: int,
        executor: Executor,
        load_fn: Callable[[], None],
        unload_fn: Callable[[], None],
    ):
        self.key = key
        self.estimated_vram_bytes = estimated_vram_bytes
        self._executor = executor
        self._load_fn = load_fn
        self._unload_fn = unload_fn
        # One lock for all residency flags; shared with the executor thread
        # (which flips _loaded), so it must be a threading.Lock, not asyncio.
        self._lock = threading.Lock()
        self._loaded = False
        self._busy = False
        self._evicting = False

    # --- introspection -----------------------------------------------------

    def is_loaded(self) -> bool:
        with self._lock:
            return self._loaded

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    # --- residency (marshalled to the model's own executor) ----------------

    async def load(self) -> None:
        """Load the model into VRAM. Idempotent. Runs the blocking build on the
        model's executor thread so it never blocks the event loop, and may raise
        (e.g. CUDA OOM) — the arbiter is responsible for making room/retrying."""
        with self._lock:
            if self._loaded:
                return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_sync)

    def _load_sync(self) -> None:
        # Executor thread. On failure _loaded stays False so accounting is honest.
        self._load_fn()
        with self._lock:
            self._loaded = True
        logger.info(f"Resident model '{self.key}' loaded")

    async def unload(self) -> None:
        """Free the model's VRAM. Precondition: caller holds an eviction
        reservation (``try_reserve_for_eviction`` returned True), which
        guarantees the model is idle so the teardown cannot race live inference.
        The reservation is cleared on completion. Idempotent-safe."""
        with self._lock:
            if not self._loaded:
                self._evicting = False
                return
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._unload_sync)
        finally:
            with self._lock:
                self._evicting = False

    def _unload_sync(self) -> None:
        # Executor thread — serialised behind any in-flight inference on this
        # same single-thread executor, which is what makes it safe (F1).
        self._unload_fn()
        with self._lock:
            self._loaded = False
        logger.info(f"Resident model '{self.key}' unloaded")

    # --- eviction / operation coordination (F3) ----------------------------

    def try_reserve_for_eviction(self) -> bool:
        """Atomically reserve this model for eviction. Succeeds only if it is
        loaded, idle, and not already being evicted. While reserved,
        ``begin_operation`` will refuse — so the subsequent unload cannot race a
        newly-started operation."""
        with self._lock:
            if not self._loaded or self._busy or self._evicting:
                return False
            self._evicting = True
            return True

    def cancel_eviction_reservation(self) -> None:
        """Release an eviction reservation without unloading (e.g. the arbiter
        decided it no longer needs to evict this one)."""
        with self._lock:
            self._evicting = False

    def begin_operation(self) -> bool:
        """Mark the start of a GPU operation. Returns False (caller must not
        proceed) if the model is not loaded or is reserved for eviction."""
        with self._lock:
            if not self._loaded or self._evicting:
                return False
            self._busy = True
            return True

    def end_operation(self) -> None:
        with self._lock:
            self._busy = False

    # --- recreation (F5) ---------------------------------------------------

    def rebind(self, executor: Executor, load_fn: Callable[[], None],
               unload_fn: Callable[[], None]) -> None:
        """Point this stable-keyed resident at a freshly-recreated processor's
        executor/model after a stuck-timeout recovery, resetting residency to
        unloaded. Keeps the arbiter's registry identity intact (no phantom
        reservation, no unaccounted reload)."""
        with self._lock:
            self._executor = executor
            self._load_fn = load_fn
            self._unload_fn = unload_fn
            self._loaded = False
            self._busy = False
            self._evicting = False
        logger.info(f"Resident model '{self.key}' rebound to a new processor instance")
