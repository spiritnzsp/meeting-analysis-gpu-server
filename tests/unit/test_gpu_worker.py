"""Tests for GPUWorker arbiter gating (D2.2): request-derived workload need,
_serve acquiring the lease + being the sole result sender, and admission-failure
handling. Uses fakes — no GPU, no real models."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from gpu_server.config import Config
from gpu_server.worker import GPUWorker
from gpu_server.orchestrator import WorkloadNeed
from gpu_server.protocol import ProcessingResult


def _options(transcribe=False, diarize=False, extract_embeddings=False):
    return SimpleNamespace(
        transcribe=transcribe,
        diarize=diarize,
        extract_embeddings=extract_embeddings,
        language=None,
        whisper_model=None,
        num_speakers=None,
    )


def _request(rid="r", **opts):
    return SimpleNamespace(
        request_id=rid, meeting_name="m", audio_data=b"",
        options=_options(**opts),
    )


class FakeLease:
    def __init__(self, arbiter):
        self._arbiter = arbiter

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._arbiter.released = True
        return False


class FakeArbiter:
    def __init__(self, raise_on_acquire=None):
        self.acquired = []
        self.released = False
        self._raise = raise_on_acquire

    async def acquire(self, need):
        if self._raise is not None:
            raise self._raise
        self.acquired.append(need)
        return FakeLease(self)


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)


@pytest.fixture
def worker_factory():
    workers = []

    def make(arbiter):
        w = GPUWorker(Config(), queue=SimpleNamespace(), arbiter=arbiter)
        workers.append(w)
        return w

    yield make
    for w in workers:
        w._whisper_handle.processor.shutdown()
        w._pyannote_handle.processor.shutdown()


def test_workload_need_is_request_derived(worker_factory):
    w = worker_factory(FakeArbiter())
    assert w._workload_need(_request(transcribe=True)).required_models == ("whisper",)
    assert w._workload_need(_request(diarize=True)).required_models == ("pyannote",)
    assert w._workload_need(
        _request(transcribe=True, diarize=True)
    ).required_models == ("whisper", "pyannote")
    # Embeddings pin the embedding model — but only alongside diarize.
    assert w._workload_need(
        _request(diarize=True, extract_embeddings=True)
    ).required_models == ("pyannote", "pyannote_embedding")
    assert w._workload_need(
        _request(extract_embeddings=True)  # no diarize → embedding not pinned
    ).required_models == ()
    assert w._workload_need(_request()).required_models == ()


async def test_serve_acquires_derived_need_and_sends_result(worker_factory):
    arb = FakeArbiter()
    w = worker_factory(arb)
    result = ProcessingResult(request_id="r", success=True, full_text="hello")

    async def fake_process(queued):
        return result

    w._process_request = fake_process

    ws = FakeWS()
    queued = SimpleNamespace(request=_request(transcribe=True, diarize=True), websocket=ws)
    await w._serve(queued)

    # Acquired exactly the derived need, and the lease was released.
    assert arb.acquired == [WorkloadNeed(required_models=("whisper", "pyannote"))]
    assert arb.released
    # Sole sender: final progress + the result JSON.
    assert len(ws.sent) == 2
    assert json.loads(ws.sent[1])["success"] is True


async def test_serve_emits_failure_result_when_acquire_raises(worker_factory):
    # F-D: acquire() can raise (unknown model, forced-load OOM, hold failure);
    # the client must get a failure result, not hang.
    arb = FakeArbiter(raise_on_acquire=KeyError("unknown model"))
    w = worker_factory(arb)

    async def fake_process(queued):  # should never run
        raise AssertionError("process must not run when acquire fails")

    w._process_request = fake_process
    ws = FakeWS()
    queued = SimpleNamespace(request=_request(transcribe=True), websocket=ws)
    await w._serve(queued)

    assert ws.sent, "a failure message must be sent"
    msg = json.loads(ws.sent[-1])
    assert msg.get("error_code") == "PROCESSING_FAILED"


async def test_serve_timeout_sends_processing_timeout(worker_factory):
    # F1: a processing timeout is reported as PROCESSING_TIMEOUT (the prior wire
    # contract), not a ProcessingResult. The processors are idle here (not wedged),
    # so no recreate happens; the lease still releases.
    arb = FakeArbiter()
    w = worker_factory(arb)
    w.config.queue.processing_timeout = 0  # force immediate timeout

    async def hang(queued):
        await asyncio.Event().wait()

    w._process_request = hang
    ws = FakeWS()
    queued = SimpleNamespace(request=_request(transcribe=True, diarize=True), websocket=ws)
    await w._serve(queued)

    assert arb.released
    assert ws.sent
    msg = json.loads(ws.sent[-1])
    assert msg.get("error_code") == "PROCESSING_TIMEOUT"
    assert msg.get("recoverable") is False
