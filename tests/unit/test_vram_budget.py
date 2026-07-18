"""Tests for VramBudget — the pure fit-arithmetic value object."""
from gpu_server.orchestrator.vram_budget import VramBudget

GB = 1024 ** 3


def _budget(total=16 * GB, free=16 * GB, resident=0, transient=0, headroom=0):
    return VramBudget(
        total_bytes=total,
        free_bytes=free,
        reserved_resident_bytes=resident,
        reserved_transient_bytes=transient,
        headroom_bytes=headroom,
    )


def test_available_is_pessimistic_min_of_free_and_accounting():
    # Accounting says 10GB free (16 - 6 reserved), driver says 8GB free -> take 8.
    b = _budget(total=16 * GB, free=8 * GB, resident=6 * GB)
    assert b.available_bytes == 8 * GB


def test_available_bound_by_accounting_when_driver_free_is_high():
    # Driver claims all free (caching allocator hides our usage), but our
    # accounting knows 13GB is reserved -> trust accounting, not the driver.
    b = _budget(total=16 * GB, free=16 * GB, resident=13 * GB)
    assert b.available_bytes == 3 * GB


def test_headroom_is_subtracted():
    b = _budget(total=16 * GB, free=16 * GB, resident=0, headroom=2 * GB)
    assert b.available_bytes == 14 * GB


def test_transient_counts_against_available():
    b = _budget(total=16 * GB, free=16 * GB, resident=2 * GB, transient=3 * GB)
    assert b.available_bytes == 11 * GB


def test_available_never_negative():
    b = _budget(total=16 * GB, free=1 * GB, resident=15 * GB, headroom=4 * GB)
    assert b.available_bytes == 0


def test_can_fit_and_shortfall_are_consistent():
    b = _budget(total=16 * GB, free=16 * GB, resident=6 * GB, headroom=1 * GB)
    # available = 9GB
    assert b.available_bytes == 9 * GB
    assert b.can_fit(9 * GB)
    assert not b.can_fit(9 * GB + 1)
    assert b.shortfall_bytes(9 * GB) == 0
    assert b.shortfall_bytes(13 * GB) == 4 * GB


def test_shortfall_zero_when_it_fits():
    b = _budget(total=16 * GB, free=16 * GB)
    assert b.shortfall_bytes(4 * GB) == 0
