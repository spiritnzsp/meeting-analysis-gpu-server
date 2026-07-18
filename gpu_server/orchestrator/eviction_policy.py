"""
Eviction policy — the *choice* of which resident models to evict under pressure.

Separated from the arbiter as a Strategy: choosing victims is the single most
likely thing to change (LRU today; size-weighted or priority-weighted on a
larger card tomorrow), and it is a pure decision with no I/O and no locks. The
arbiter owns the invariant and the lock; it delegates the choice here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class EvictionCandidate:
    """The minimal facts a policy needs about a loaded, evictable resident.

    Deliberately NOT the model itself: the policy should not touch VRAM or know
    how a model loads. ``last_used`` is scheduling state owned by the arbiter's
    registry, surfaced here only as a plain number.
    """

    key: str
    estimated_vram_bytes: int
    last_used: float


class EvictionPolicy(Protocol):
    """Picks victims to free at least ``shortfall_bytes`` from the candidates."""

    def select_victims(
        self, candidates: Sequence[EvictionCandidate], shortfall_bytes: int
    ) -> list[str]:
        ...


class LruEvictionPolicy:
    """Evict least-recently-used first until the shortfall is covered.

    Stops as soon as enough estimated VRAM has been selected, so it frees the
    fewest, oldest models needed — not everything. Returns victim keys in
    eviction order (oldest first).
    """

    def select_victims(
        self, candidates: Sequence[EvictionCandidate], shortfall_bytes: int
    ) -> list[str]:
        if shortfall_bytes <= 0:
            return []
        victims: list[str] = []
        freed = 0
        # Oldest last_used first. Stable so equal timestamps keep input order.
        for candidate in sorted(candidates, key=lambda c: c.last_used):
            if freed >= shortfall_bytes:
                break
            victims.append(candidate.key)
            freed += candidate.estimated_vram_bytes
        return victims
