"""Tests for LlmWorker request handling and server LLM wiring (fakes, no GPU)."""
import asyncio
import json

from gpu_server.config import Config
from gpu_server.llm_worker import LlmWorker
from gpu_server.processors import ProcessorCancelled
from gpu_server.protocol import LlmGenerateRequest
from gpu_server.server import GPUServer


class FakeLease:
    def __init__(self):
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False


class FakeArbiter:
    def __init__(self):
        self.acquired = []
        self.lease = FakeLease()

    async def acquire(self, need):
        self.acquired.append(need)
        return self.lease


class FakeProcessor:
    def __init__(self, text="RESULT", finish="stop", raises=None):
        self._text = text
        self._finish = finish
        self._raises = raises
        self.is_processing = False
        self.cancelled_with = None

    async def generate(self, **kwargs):
        if self._raises:
            raise self._raises
        return self._text, self._finish

    def cancel(self, request_id):
        self.cancelled_with = request_id


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)


class SlowProcessor:
    """Generation that runs until cancelled — to exercise the timeout/drain path."""

    def __init__(self):
        self.is_processing = False
        self._cancelled = False
        self.cancel_called_with = None

    async def generate(self, request_id="", **kwargs):
        for _ in range(10000):
            if self._cancelled:
                raise ProcessorCancelled("cancelled")
            await asyncio.sleep(0.01)
        return "done", "stop"

    def cancel(self, request_id):
        self.cancel_called_with = request_id
        self._cancelled = True


class FakeQueued:
    def __init__(self, request, websocket):
        self.request = request
        self.websocket = websocket
        self.cancelled = False


def _worker(processor):
    w = LlmWorker(Config(), queue=None, arbiter=FakeArbiter())
    w._processor = processor
    return w


async def test_process_success_sends_llm_result():
    worker = _worker(FakeProcessor(text="hello world", finish="stop"))
    ws = FakeWS()
    req = LlmGenerateRequest(request_id="r1", system_prompt="s", user_prompt="u")
    await worker._process(FakeQueued(req, ws))

    result = json.loads(ws.sent[0])
    assert result["type"] == "llm_result"
    assert result["success"] is True
    assert result["text"] == "hello world"
    assert result["finish_reason"] == "stop"
    # Acquired a lease requiring the LLM model, and released it.
    assert worker._arbiter.acquired[0].required_models == ("llm",)
    assert worker._arbiter.lease.exited is True


async def test_process_failure_sends_unsuccessful_result():
    worker = _worker(FakeProcessor(raises=RuntimeError("boom")))
    ws = FakeWS()
    req = LlmGenerateRequest(request_id="r2", system_prompt="s", user_prompt="u")
    await worker._process(FakeQueued(req, ws))

    result = json.loads(ws.sent[0])
    assert result["success"] is False
    assert "boom" in result["error_message"]
    assert worker._arbiter.lease.exited is True  # lease released even on failure


async def test_acquire_reserves_kv_headroom_as_transient():
    # C8: the compute-scratch margin is passed as WorkloadNeed.transient_bytes.
    cfg = Config()
    worker = LlmWorker(cfg, queue=None, arbiter=FakeArbiter())
    worker._processor = FakeProcessor()
    await worker._process(FakeQueued(LlmGenerateRequest("r", "s", "u"), FakeWS()))
    need = worker._arbiter.acquired[0]
    assert need.required_models == ("llm",)
    assert need.transient_bytes == cfg.llm.kv_headroom_bytes


async def test_timeout_cancels_drains_and_releases_lease():
    # C2/C6: a timed-out generation is actually cancelled and drained (still
    # holding the lease) before the worker moves on.
    cfg = Config()
    cfg.llm_queue.processing_timeout = 0.05
    worker = LlmWorker(cfg, queue=None, arbiter=FakeArbiter())
    proc = SlowProcessor()
    worker._processor = proc
    ws = FakeWS()
    await worker._process(FakeQueued(LlmGenerateRequest("r1", "s", "u"), ws))

    result = json.loads(ws.sent[0])
    assert result["success"] is False
    assert "timed out" in result["error_message"]
    assert proc.cancel_called_with == "r1"          # cancel was signalled
    assert proc._cancelled                           # generation drained via cancel
    assert worker._arbiter.lease.exited is True      # lease released only after drain


async def test_processor_cancelled_reports_cancelled_result():
    worker = _worker(FakeProcessor(raises=ProcessorCancelled("stop")))
    ws = FakeWS()
    await worker._process(FakeQueued(LlmGenerateRequest("r", "s", "u"), ws))
    result = json.loads(ws.sent[0])
    assert result["success"] is False
    assert "cancel" in result["error_message"].lower()


async def test_server_rejects_llm_when_disabled():
    srv = GPUServer(Config())  # llm disabled
    ws = FakeWS()
    await srv._handle_llm_request(ws, {"request_id": "r", "user_prompt": "hi"})
    msg = json.loads(ws.sent[0])
    assert msg["error_code"] == "LLM_NOT_ENABLED"


def test_server_has_no_llm_stack_by_default():
    srv = GPUServer(Config())
    assert srv.llm_worker is None
    assert srv.arbiter is None
    assert srv.llm_queue is None
    assert srv._build_workloads()["llm"] is False


def test_server_builds_llm_stack_when_enabled():
    cfg = Config()
    cfg.llm.enabled = True
    srv = GPUServer(cfg)
    assert srv.llm_worker is not None
    assert srv.arbiter is not None
    assert srv.llm_queue is not None
    caps = srv._build_workloads()
    assert caps["llm"] is True
    assert caps["transcribe"] is True
