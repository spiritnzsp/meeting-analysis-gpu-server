"""Tests for VideoWorker arbiter gating (D2.3): the transient-VRAM lease is
acquired with the right need and RELEASED before result delivery/streaming, and a
timeout kills the encoder while still releasing the lease. Uses fakes — no ffmpeg."""
from types import SimpleNamespace

import pytest

from gpu_server.config import Config
from gpu_server.video_worker import VideoWorker
from gpu_server.orchestrator import WorkloadNeed


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)


def _make_arbiter(events):
    class FakeLease:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            events.append("lease_released")
            return False

    class FakeArbiter:
        def __init__(self):
            self.need = None

        async def acquire(self, need):
            self.need = need
            events.append("acquired")
            return FakeLease()

    return FakeArbiter()


class FakeEncoder:
    def __init__(self, success=True, hang=False):
        self._success = success
        self._hang = hang
        self.cancelled = None
        self.idle_waited = False

    async def encode(self, input_path, output_path, options, progress_callback, request_id):
        if self._hang:
            import asyncio
            await asyncio.Event().wait()  # never returns until cancelled
        return (self._success, None if self._success else "boom", "h264_nvenc")

    def cancel(self, request_id=""):
        self.cancelled = request_id

    async def wait_for_idle(self, timeout=30.0):
        self.idle_waited = True
        return True


def _worker(events):
    cfg = Config()
    cfg.video_encoding.enabled = True
    w = VideoWorker(cfg, queue=SimpleNamespace(), arbiter=_make_arbiter(events))
    return w


def _shared_fs_request(tmp_path):
    inp = tmp_path / "in.mp4"
    inp.write_bytes(b"data")
    return SimpleNamespace(
        request_id="v1",
        filename="in.mp4",
        transfer_method="shared_fs",
        input_path=str(inp),
        output_path=str(tmp_path / "out.mp4"),
        options=SimpleNamespace(),
    )


async def test_lease_released_before_delivery(tmp_path):
    events = []
    w = _worker(events)
    w._encoder = FakeEncoder(success=True)

    # Spy on delivery so we can assert it runs AFTER the lease releases.
    async def spy_deliver(queued, outcome, output_temp, is_shared_fs):
        events.append("deliver")

    w._deliver = spy_deliver

    ws = FakeWS()
    queued = SimpleNamespace(request=_shared_fs_request(tmp_path), websocket=ws)
    await w._serve(queued)

    # Transient-VRAM need, no required models.
    assert w._arbiter.need == WorkloadNeed(
        required_models=(),
        transient_bytes=w.config.video_encoding.per_session_vram_bytes,
    )
    # P2-2: the lease is released BEFORE delivery/streaming.
    assert events == ["acquired", "lease_released", "deliver"]


async def test_timeout_kills_encoder_and_releases_lease(tmp_path):
    events = []
    w = _worker(events)
    w._encoder = FakeEncoder(hang=True)
    w.config.video_queue.processing_timeout = 0  # force immediate timeout

    ws = FakeWS()
    queued = SimpleNamespace(request=_shared_fs_request(tmp_path), websocket=ws)
    await w._serve(queued)

    assert w._encoder.cancelled == "v1"      # ffmpeg killed
    assert w._encoder.idle_waited            # waited for the kill to settle (F7)
    assert "lease_released" in events        # lease always released
    # A timeout error was sent to the client.
    assert any("PROCESSING_TIMEOUT" in m for m in ws.sent)


async def test_encode_failure_sends_result_and_releases_lease(tmp_path):
    events = []
    w = _worker(events)
    w._encoder = FakeEncoder(success=False)

    ws = FakeWS()
    queued = SimpleNamespace(request=_shared_fs_request(tmp_path), websocket=ws)
    await w._serve(queued)

    assert "lease_released" in events
    assert ws.sent  # a failure VideoEncodeResult was sent
