"""
Resident processor handle — owns a GPU processor's recreation and the rebinding
of its arbiter residents as ONE transaction.

A stuck-timeout recovery must swap the processor object *safely behind the
arbiter's back*: the arbiter holds ``ResidentModel`` objects keyed by a stable
key, and those residents' load/unload callables are bound to the OLD processor's
executor. If recovery simply built a new processor, the arbiter would keep
calling the dead executor and its accounting would desync (Phase-A concurrency
review, F5). This handle sequences it correctly:

  1. quiesce the old processor OFF the event loop — its ``shutdown()`` does a
     blocking drain-wait; running it on the loop would freeze every other
     workload for up to the drain timeout (P1-1),
  2. build a fresh processor via the factory,
  3. ``rebind`` EACH of the processor's residents (all of them — e.g. both
     pyannote keys) to the new executor and freshly-bound load/unload callables,
     as one step, so the registry never observes a half-swapped processor.

Ownership rationale: NOT the arbiter (it must never touch CUDA/executors) and
NOT the bare worker (which would then reach into processor internals). The handle
is the seam that owns "a processor and its residents move together".
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Executor
from typing import List, Optional, Protocol, runtime_checkable

from ..logging_config import get_logger
from .resident_model import ResidentBinding, ResidentModel

logger = get_logger(__name__)


@runtime_checkable
class ResidentCapable(Protocol):
    """A processor the handle can manage: it exposes its residents and its
    executor, and can be shut down. Structural, so the handle stays decoupled
    from the concrete processor classes."""

    @property
    def executor(self) -> Optional[Executor]: ...
    def resident_bindings(self) -> List[ResidentBinding]: ...
    def shutdown(self, timeout: float = ...) -> None: ...


class ResidentProcessorHandle:
    """Owns one processor instance and the ``ResidentModel``s built from it.

    The residents are created ONCE (stable identity) and handed to the arbiter's
    registry; ``recreate()`` rebinds those same objects onto a fresh processor.
    """

    def __init__(self, factory: Callable[[], ResidentCapable]):
        self._factory = factory
        self._processor = factory()
        # Build residents once; the arbiter registers exactly these objects and
        # relies on their identity surviving a recreate (rebind, not re-register).
        self._residents: dict[str, ResidentModel] = {}
        for binding in self._processor.resident_bindings():
            if binding.key in self._residents:
                raise ValueError(
                    f"duplicate resident key '{binding.key}' from "
                    f"{type(self._processor).__name__}"
                )
            self._residents[binding.key] = ResidentModel(
                key=binding.key,
                estimated_vram_bytes=binding.estimated_vram_bytes,
                executor=self._processor.executor,
                load_fn=binding.load_fn,
                unload_fn=binding.unload_fn,
            )

    @property
    def processor(self) -> ResidentCapable:
        """The current (live) processor instance."""
        return self._processor

    def residents(self) -> List[ResidentModel]:
        """The stable-identity residents for the arbiter to register."""
        return list(self._residents.values())

    async def recreate(self) -> None:
        """Quiesce the old processor and rebind every resident onto a fresh one.

        MUST be called while the caller still holds an arbiter lease on this
        processor's models (op_count > 0), so no concurrent admission can reserve
        or evict a model mid-swap. The shutdown runs off the event loop; the
        rebind is synchronous per resident (each under the resident's own lock).
        """
        old = self._processor
        loop = asyncio.get_running_loop()
        # P1-1: shutdown() blocks on the executor drain (up to its timeout). Run
        # it off the loop so a stuck drain cannot freeze the other workloads.
        await loop.run_in_executor(None, old.shutdown)

        new = self._factory()
        # From here on `new` owns a live executor thread. If anything fails before
        # the swap completes, shut it down so a failed recovery doesn't strand a
        # GPU-executor thread (F2). The handle is then left pointing at the
        # already-shut-down `old` and is spent — the failure is a "can't happen"
        # programming error (same class ⇒ same residents), surfaced loudly.
        try:
            bindings = {b.key: b for b in new.resident_bindings()}
            missing = set(self._residents) - set(bindings)
            if missing:
                raise KeyError(
                    f"recreated {type(new).__name__} is missing residents: {sorted(missing)}"
                )
            for key, resident in self._residents.items():
                binding = bindings[key]
                resident.rebind(new.executor, binding.load_fn, binding.unload_fn)
        except BaseException:
            new.shutdown()
            raise
        self._processor = new
        logger.info(
            f"Recreated {type(new).__name__} and rebound residents: "
            f"{sorted(self._residents)}"
        )
