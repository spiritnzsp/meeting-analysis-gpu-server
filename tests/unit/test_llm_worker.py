"""Tests for LlmWorker request handling and server LLM wiring (fakes, no GPU)."""
import json

from gpu_server.config import Config
from gpu_server.llm_worker import LlmWorker
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
