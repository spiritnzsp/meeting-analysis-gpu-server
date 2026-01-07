"""
Tests for error message classes.
"""
import json
import pytest

from gpu_server.protocol import (
    ErrorMessage,
    AuthErrorMessage,
    CancelledMessage,
    MessageType,
)


class TestErrorMessage:
    """Tests for ErrorMessage class."""

    def test_basic_error_message(self):
        """Basic error message should serialize correctly."""
        msg = ErrorMessage(
            request_id="req-123",
            error="Something went wrong",
        )
        data = json.loads(msg.to_json())

        assert data["type"] == MessageType.ERROR
        assert data["request_id"] == "req-123"
        assert data["error"] == "Something went wrong"
        assert data["recoverable"] is True  # Default

    def test_error_message_with_error_code(self):
        """Error message with error_code should include it in JSON."""
        msg = ErrorMessage(
            request_id="req-456",
            error="Validation failed",
            error_code="VALIDATION_ERROR",
            recoverable=False,
        )
        data = json.loads(msg.to_json())

        assert data["type"] == MessageType.ERROR
        assert data["request_id"] == "req-456"
        assert data["error"] == "Validation failed"
        assert data["error_code"] == "VALIDATION_ERROR"
        assert data["recoverable"] is False

    def test_error_message_without_error_code(self):
        """Error message without error_code should not include it."""
        msg = ErrorMessage(
            request_id="req-789",
            error="Generic error",
        )
        data = json.loads(msg.to_json())

        assert "error_code" not in data

    def test_error_message_recoverable_false(self):
        """Non-recoverable error message should serialize correctly."""
        msg = ErrorMessage(
            request_id="req-000",
            error="Fatal error",
            recoverable=False,
        )
        data = json.loads(msg.to_json())

        assert data["recoverable"] is False


class TestAuthErrorMessage:
    """Tests for AuthErrorMessage class."""

    def test_basic_auth_error(self):
        """Basic auth error should serialize correctly."""
        msg = AuthErrorMessage(
            error="Invalid API key",
        )
        data = json.loads(msg.to_json())

        assert data["type"] == MessageType.AUTH_FAILED
        assert data["error"] == "Invalid API key"
        assert data["error_code"] == "AUTH_ERROR"  # Default

    def test_auth_error_with_custom_code(self):
        """Auth error with custom error_code should serialize correctly."""
        msg = AuthErrorMessage(
            error="Connection limit reached",
            error_code="CONNECTION_LIMIT",
        )
        data = json.loads(msg.to_json())

        assert data["error"] == "Connection limit reached"
        assert data["error_code"] == "CONNECTION_LIMIT"

    def test_auth_error_with_protocol_version(self):
        """Auth error with server_protocol_version should include it."""
        msg = AuthErrorMessage(
            error="Protocol version mismatch",
            error_code="PROTOCOL_VERSION_MISMATCH",
            server_protocol_version="1.0",
        )
        data = json.loads(msg.to_json())

        assert data["error_code"] == "PROTOCOL_VERSION_MISMATCH"
        assert data["server_protocol_version"] == "1.0"

    def test_auth_error_without_protocol_version(self):
        """Auth error without protocol_version should not include it."""
        msg = AuthErrorMessage(
            error="Auth timeout",
            error_code="AUTH_TIMEOUT",
        )
        data = json.loads(msg.to_json())

        assert "server_protocol_version" not in data

    def test_auth_error_no_request_id(self):
        """Auth error should not have request_id field."""
        msg = AuthErrorMessage(error="Some auth error")
        data = json.loads(msg.to_json())

        assert "request_id" not in data


class TestCancelledMessage:
    """Tests for CancelledMessage class."""

    def test_cancelled_message(self):
        """Cancelled message should serialize correctly."""
        msg = CancelledMessage(request_id="req-cancel-123")
        data = json.loads(msg.to_json())

        assert data["type"] == MessageType.CANCELLED
        assert data["request_id"] == "req-cancel-123"

    def test_cancelled_message_only_has_required_fields(self):
        """Cancelled message should only have type and request_id."""
        msg = CancelledMessage(request_id="req-456")
        data = json.loads(msg.to_json())

        assert set(data.keys()) == {"type", "request_id"}


class TestErrorCodes:
    """Tests for standardized error codes."""

    def test_all_error_codes_are_uppercase(self):
        """All error codes should follow UPPERCASE_SNAKE_CASE convention."""
        error_codes = [
            "VALIDATION_ERROR",
            "RATE_LIMIT_EXCEEDED",
            "QUEUE_FULL",
            "ENQUEUE_FAILED",
            "INTERNAL_ERROR",
            "UNKNOWN_MESSAGE_TYPE",
            "CANCEL_NOT_FOUND",
            "AUTH_ERROR",
            "AUTH_TIMEOUT",
            "INVALID_API_KEY",
            "PROTOCOL_VERSION_MISMATCH",
            "CONNECTION_LIMIT",
            "QUEUE_TIMEOUT",
            "PROCESSING_TIMEOUT",
        ]

        for code in error_codes:
            # Verify format: UPPERCASE with underscores
            assert code == code.upper(), f"{code} should be uppercase"
            assert " " not in code, f"{code} should not contain spaces"
            assert code.replace("_", "").isalnum(), f"{code} should be alphanumeric with underscores"
