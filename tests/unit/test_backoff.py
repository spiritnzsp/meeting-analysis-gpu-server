"""Tests for ErrorBackoff — the shared exponential worker-loop backoff (D3)."""
import asyncio

import pytest

from gpu_server.backoff import ErrorBackoff


async def test_backoff_doubles_and_caps(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    b = ErrorBackoff(initial=1.0, maximum=8.0)
    for _ in range(6):
        await b.sleep()
    assert slept == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


async def test_backoff_reset(monkeypatch):
    async def fake_sleep(s):
        pass

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    b = ErrorBackoff(initial=1.0, maximum=8.0)
    await b.sleep()
    await b.sleep()
    assert b.current_seconds == 4.0
    b.reset()
    assert b.current_seconds == 1.0
