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
from dataclasses import dataclass
from typing import Callable, Protocol

from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResidentBinding:
    """One resident a processor owns: its stable key, its VRAM size, and the
    blocking load/unload callables (already bound to the processor instance).

    Produced by processors (``resident_bindings()``) and consumed by the arbiter
    side (``ResidentProcessorHandle`` turns each into a ``ResidentModel``). It
    lives here beside ``ResidentModel`` — the residency vocabulary — so a
    processor declaring its residents does not have to import the handle
    (consumer) module. The processor's own executor is supplied separately, since
    a single processor's residents all share its one executor."""

    key: str
    estimated_vram_bytes: int
    load_fn: Callable[[], None]
    unload_fn: Callable[[], None]


class ManagedModel(Protocol):
    """Residency contract the arbiter depends on. Intentionally slim: only what
    it takes to load, unload, and size a model. Scheduling state (last_used) and
    the choice to evict live in the arbiter, not here."""

    key: str
    estimated_vram_bytes: int
    is_busy: bool  # whether an operation/lease currently holds this model

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
        # Refcount, not a bool: more than one lease can legitimately hold the
        # same resident model at once, so residency must survive until the LAST
        # holder ends. A bool would let the first release make a still-in-use
        # model evictable.
        self._op_count = 0
        self._evicting = False
        # Serializes concurrent load() calls so a check-then-dispatch race can't
        # build the model twice on the executor (S4).
        self._load_lock = asyncio.Lock()

    # --- introspection -----------------------------------------------------

    def is_loaded(self) -> bool:
        with self._lock:
            return self._loaded

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._op_count > 0

    # --- residency (marshalled to the model's own executor) ----------------

    async def load(self) -> None:
        """Load the model into VRAM. Idempotent (including under concurrent
        callers). Runs the blocking build on the model's executor thread so it
        never blocks the event loop, and may raise (e.g. CUDA OOM) — the arbiter
        is responsible for making room/retrying."""
        with self._lock:
            if self._loaded:
                return
        async with self._load_lock:
            # Re-check under the load lock: a racing caller may have loaded it
            # while we waited (S4).
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
        """Free the model's VRAM. PRECONDITION: the caller holds an eviction
        reservation (``try_reserve_for_eviction`` returned True), which
        guarantees the model is idle so the teardown cannot race live inference.
        Enforced, not merely assumed (S2): unloading a model that is not
        reserved-for-eviction raises, because for such a call the executor
        backstop only serializes the one in-flight inference, not the job's
        continued need for residency. The reservation is cleared on completion."""
        with self._lock:
            if not self._loaded:
                self._evicting = False
                return
            if not self._evicting:
                raise RuntimeError(
                    f"unload('{self.key}') requires an eviction reservation; "
                    "call try_reserve_for_eviction() first"
                )
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
        loaded, idle (no operation in flight), and not already being evicted.
        While reserved, ``begin_operation`` will refuse — so the subsequent
        unload cannot race a newly-started operation."""
        with self._lock:
            if not self._loaded or self._op_count > 0 or self._evicting:
                return False
            self._evicting = True
            return True

    def cancel_eviction_reservation(self) -> None:
        """Release an eviction reservation without unloading (e.g. the arbiter
        decided it no longer needs to evict this one)."""
        with self._lock:
            self._evicting = False

    def begin_operation(self) -> bool:
        """Mark the start of a use of this model (increments the hold count).
        Returns False (caller must not proceed) if the model is not loaded or is
        reserved for eviction. Balanced by ``end_operation``."""
        with self._lock:
            if not self._loaded or self._evicting:
                return False
            self._op_count += 1
            return True

    def end_operation(self) -> None:
        """End one use of this model (decrements the hold count). The model
        becomes evictable again only when the count reaches zero."""
        with self._lock:
            if self._op_count > 0:
                self._op_count -= 1

    # --- recreation (F5) ---------------------------------------------------

    def rebind(self, executor: Executor, load_fn: Callable[[], None],
               unload_fn: Callable[[], None]) -> None:
        """Point this stable-keyed resident at a freshly-recreated processor's
        executor/model after a stuck-timeout recovery, resetting residency to
        unloaded. Keeps the arbiter's registry identity intact (no phantom
        reservation, no unaccounted reload).

        PRECONDITION: the model must be quiesced — no ``load``/``unload`` in
        flight on the OLD executor. The stuck-timeout recovery guarantees this
        by shutting the old processor down first; otherwise a `_load_sync` still
        queued on the old executor could later flip `_loaded` back on and leave
        a phantom-loaded model whose new ``load_fn`` never ran (S5)."""
        with self._lock:
            self._executor = executor
            self._load_fn = load_fn
            self._unload_fn = unload_fn
            self._loaded = False
            self._op_count = 0
            self._evicting = False
        logger.info(f"Resident model '{self.key}' rebound to a new processor instance")
