"""Tests for the server message RequestRouter (D4) and LLM cancel parity (C4)."""
import json

import pytest

from gpu_server.config import Config
from gpu_server.protocol import MessageType
from gpu_server.server import GPUServer


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)


class FakeQueue:
    def __init__(self, hit):
        self._hit = hit
        self.asked = None

    async def cancel(self, request_id):
        self.asked = request_id
        return self._hit


def _server(**cfg_over):
    cfg = Config()
    for k, v in cfg_over.items():
        setattr(cfg.llm, k, v) if hasattr(cfg.llm, k) else None
    srv = GPUServer(cfg)
    return srv


def _shutdown(srv):
    srv.worker._whisper_handle.processor.shutdown()
    srv.worker._pyannote_handle.processor.shutdown()
    if srv.llm_worker:
        srv.llm_worker._processor.shutdown()


def test_router_map_has_expected_handlers():
    srv = GPUServer(Config())
    try:
        assert set(srv._handlers) == {
            MessageType.PROCESS, MessageType.VIDEO_ENCODE, MessageType.LLM_GENERATE,
            MessageType.CANCEL, MessageType.PING, "auth",
        }
    finally:
        _shutdown(srv)


async def test_ping_handler_reports_queue_state():
    srv = GPUServer(Config())
    try:
        ws = FakeWS()
        await srv._handle_ping(ws, {})
        msg = json.loads(ws.sent[0])
        assert msg["type"] == MessageType.PONG
        assert "queue_size" in msg and "is_processing" in msg
    finally:
        _shutdown(srv)


async def test_cancel_falls_through_to_llm_queue():
    # C4: a queued LLM generation must be cancellable at parity with audio/video.
    cfg = Config()
    cfg.llm.enabled = True
    cfg.llm.model_path = "x"
    srv = GPUServer(cfg)
    try:
        srv.queue = FakeQueue(hit=False)      # audio: not found
        srv.llm_queue = FakeQueue(hit=True)   # llm: found here
        ws = FakeWS()
        await srv._handle_cancel(ws, {"request_id": "r"})
        assert srv.llm_queue.asked == "r"
        msg = json.loads(ws.sent[0])
        # Success path → CancelledMessage, not a CANCEL_NOT_FOUND error.
        assert msg.get("error_code") != "CANCEL_NOT_FOUND"
    finally:
        _shutdown(srv)


async def test_cancel_not_found_when_no_queue_hits():
    srv = GPUServer(Config())  # llm disabled → no llm_queue
    try:
        srv.queue = FakeQueue(hit=False)
        ws = FakeWS()
        await srv._handle_cancel(ws, {"request_id": "missing"})
        msg = json.loads(ws.sent[0])
        assert msg.get("error_code") == "CANCEL_NOT_FOUND"
    finally:
        _shutdown(srv)
