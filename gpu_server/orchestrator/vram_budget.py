"""
VRAM budget arithmetic.

A pure value object: given a memory snapshot and our own accounting of what is
loaded/reserved, it answers "does this fit?" and "how much must I free?". No
I/O, no torch, no locks — so the crux of admission is trivially unit-testable in
isolation. All scheduling, eviction, and thread coordination live elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VramBudget:
    """Immutable snapshot of the VRAM situation for one admission decision.

    Two independent views of "how much can I still use" are reconciled
    pessimistically:

    * ``free_bytes`` — the driver's live free figure. Reflects real usage
      including other processes and NVENC, but is a lower bound (torch's caching
      allocator hides freed-but-reserved blocks) and can drift.
    * ``total - reserved_resident - reserved_transient`` — our own accounting
      from model VRAM estimates and active transient leases.

    ``available_bytes`` takes the MINIMUM of the two (minus a safety headroom) so
    we never over-admit against either blind spot. This can under-admit (leave
    capacity idle) — a deliberate trade of utilisation for OOM-safety.
    """

    total_bytes: int
    free_bytes: int
    reserved_resident_bytes: int
    reserved_transient_bytes: int
    headroom_bytes: int

    @property
    def available_bytes(self) -> int:
        by_accounting = (
            self.total_bytes
            - self.reserved_resident_bytes
            - self.reserved_transient_bytes
        )
        pessimistic = min(self.free_bytes, by_accounting)
        return max(0, pessimistic - self.headroom_bytes)

    def can_fit(self, need_bytes: int) -> bool:
        """Whether ``need_bytes`` fits without oversubscribing (with headroom)."""
        return need_bytes <= self.available_bytes

    def shortfall_bytes(self, need_bytes: int) -> int:
        """How many bytes must be freed for ``need_bytes`` to fit (0 if it already does)."""
        return max(0, need_bytes - self.available_bytes)
