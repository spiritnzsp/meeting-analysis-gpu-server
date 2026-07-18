"""Tests for the v1.2 LLM protocol messages."""
import json

import pytest

from gpu_server.protocol import (
    PROTOCOL_VERSION,
    LlmGenerateRequest,
    LlmGenerateResult,
    MessageType,
)
from gpu_server.validation import ValidationError


def test_protocol_bumped_to_1_2():
    assert PROTOCOL_VERSION == (1, 2)


def test_message_types_present():
    assert MessageType.LLM_GENERATE == "llm_generate"
    assert MessageType.LLM_RESULT == "llm_result"


def test_request_round_trip():
    req = LlmGenerateRequest(
        request_id="r1",
        system_prompt="SYS",
        user_prompt="USER",
        temperature=0.3,
        max_tokens=1000,
        response_format="json_object",
        priority=1,
        meeting_name="Standup",
    )
    data = json.loads(req.to_json())
    assert data["type"] == "llm_generate"
    back = LlmGenerateRequest.from_dict(data)
    assert back == req


def test_request_defaults_are_optional():
    req = LlmGenerateRequest.from_dict(
        {"request_id": "r", "user_prompt": "hi"}
    )
    assert req.system_prompt == ""
    assert req.temperature is None
    assert req.max_tokens is None
    assert req.response_format is None


def test_request_rejects_missing_user_prompt():
    with pytest.raises(ValidationError):
        LlmGenerateRequest.from_dict({"request_id": "r", "user_prompt": ""})


def test_request_rejects_missing_request_id():
    with pytest.raises(ValidationError):
        LlmGenerateRequest.from_dict({"user_prompt": "hi"})


def test_request_rejects_bad_max_tokens():
    with pytest.raises(ValidationError):
        LlmGenerateRequest.from_dict(
            {"request_id": "r", "user_prompt": "hi", "max_tokens": -5}
        )


def test_request_rejects_bad_temperature():
    with pytest.raises(ValidationError):
        LlmGenerateRequest.from_dict(
            {"request_id": "r", "user_prompt": "hi", "temperature": "hot"}
        )


def test_request_rejects_bad_response_format():
    with pytest.raises(ValidationError):
        LlmGenerateRequest.from_dict(
            {"request_id": "r", "user_prompt": "hi", "response_format": "yaml"}
        )
    # None and "json_object" are accepted
    assert LlmGenerateRequest.from_dict(
        {"request_id": "r", "user_prompt": "hi", "response_format": "json_object"}
    ).response_format == "json_object"


def test_request_rejects_out_of_range_temperature():
    with pytest.raises(ValidationError):
        LlmGenerateRequest.from_dict(
            {"request_id": "r", "user_prompt": "hi", "temperature": 5.0}
        )
    with pytest.raises(ValidationError):
        LlmGenerateRequest.from_dict(
            {"request_id": "r", "user_prompt": "hi", "temperature": -1.0}
        )


def test_request_rejects_oversized_prompt():
    from gpu_server.protocol import MAX_LLM_PROMPT_CHARS
    with pytest.raises(ValidationError):
        LlmGenerateRequest.from_dict(
            {"request_id": "r", "user_prompt": "x" * (MAX_LLM_PROMPT_CHARS + 1)}
        )


def test_result_round_trip():
    res = LlmGenerateResult(
        request_id="r1", success=True, text='{"ok": true}',
        finish_reason="stop", processing_time_seconds=1.5,
    )
    data = json.loads(res.to_json())
    assert data["type"] == "llm_result"
    assert LlmGenerateResult.from_dict(data) == res
