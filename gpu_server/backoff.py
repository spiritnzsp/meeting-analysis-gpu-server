"""
Exponential error backoff — the one piece genuinely shared by all three worker
drain loops (audio, video, LLM). Each loop resets it after a clean iteration and
sleeps it after an unexpected error, so a persistent failure can't flood the log.

Only this is extracted: the loops' dequeue + queue-cancel handling genuinely
diverge per worker (audio sends an error, video cleans the pending upload, LLM
just continues), so a shared "_next" would need per-worker callbacks that cost
more than the duplication they remove.
"""
from __future__ import annotations

import asyncio


class ErrorBackoff:
    """Monotonic exponential backoff: initial → doubling → capped at maximum."""

    def __init__(self, initial: float = 1.0, maximum: float = 60.0):
        self._initial = initial
        self._maximum = maximum
        self._current = initial

    def reset(self) -> None:
        """Call after a successful loop iteration."""
        self._current = self._initial

    async def sleep(self) -> None:
        """Sleep the current interval, then double it (capped)."""
        await asyncio.sleep(self._current)
        self._current = min(self._current * 2, self._maximum)

    @property
    def current_seconds(self) -> float:
        return self._current
