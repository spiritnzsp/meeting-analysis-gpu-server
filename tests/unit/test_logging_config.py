"""
Tests for the logging configuration module.
"""
import json
import logging
import pytest

from gpu_server.logging_config import (
    LogContext,
    set_request_context,
    clear_request_context,
    StructuredFormatter,
    ConsoleFormatter,
    ContextLogger,
    get_logger,
    configure_logging,
    PerformanceMetrics,
    PerformanceTimer,
    LogEvents,
    request_id_var,
    client_addr_var,
    meeting_name_var,
)


class TestLogContext:
    """Tests for LogContext."""

    def test_current_returns_context(self):
        """LogContext.current() should return current context values."""
        clear_request_context()
        set_request_context(
            request_id="test-123",
            client_addr="192.168.1.1",
            meeting_name="Test Meeting",
        )

        ctx = LogContext.current()

        assert ctx.request_id == "test-123"
        assert ctx.client_addr == "192.168.1.1"
        assert ctx.meeting_name == "Test Meeting"

        clear_request_context()

    def test_current_returns_empty_when_not_set(self):
        """LogContext.current() should return empty strings when not set."""
        clear_request_context()

        ctx = LogContext.current()

        assert ctx.request_id == ""
        assert ctx.client_addr == ""
        assert ctx.meeting_name == ""


class TestRequestContext:
    """Tests for request context management."""

    def test_set_request_context(self):
        """set_request_context should set context variables."""
        clear_request_context()

        set_request_context(request_id="req-456")

        assert request_id_var.get() == "req-456"

        clear_request_context()

    def test_set_partial_context(self):
        """set_request_context should allow partial updates."""
        clear_request_context()

        set_request_context(request_id="req-789")
        set_request_context(client_addr="10.0.0.1")

        assert request_id_var.get() == "req-789"
        assert client_addr_var.get() == "10.0.0.1"

        clear_request_context()

    def test_clear_request_context(self):
        """clear_request_context should reset all context variables."""
        set_request_context(
            request_id="test",
            client_addr="addr",
            meeting_name="meeting",
        )

        clear_request_context()

        assert request_id_var.get() == ""
        assert client_addr_var.get() == ""
        assert meeting_name_var.get() == ""


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_formats_as_json(self):
        """StructuredFormatter should output valid JSON."""
        formatter = StructuredFormatter(include_context=False)
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert "timestamp" in data

    def test_includes_context_when_set(self):
        """StructuredFormatter should include context when available."""
        clear_request_context()
        set_request_context(request_id="ctx-req-123")

        formatter = StructuredFormatter(include_context=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="With context",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        data = json.loads(output)

        assert data["request_id"] == "ctx-req-123"

        clear_request_context()

    def test_includes_location_for_warnings(self):
        """StructuredFormatter should include location for WARNING and above."""
        formatter = StructuredFormatter(include_context=False)
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="/path/to/file.py",
            lineno=100,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.funcName = "test_function"

        output = formatter.format(record)
        data = json.loads(output)

        assert "location" in data
        assert data["location"]["line"] == 100
        assert data["location"]["function"] == "test_function"


class TestConsoleFormatter:
    """Tests for ConsoleFormatter."""

    def test_formats_with_timestamp(self):
        """ConsoleFormatter output should include timestamp."""
        formatter = ConsoleFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Console message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "test.module" in output
        assert "Console message" in output
        # Should have a timestamp-like pattern (HH:MM:SS)
        assert ":" in output


class TestContextLogger:
    """Tests for ContextLogger."""

    def test_get_logger_returns_context_logger(self):
        """get_logger should return a ContextLogger instance."""
        logger = get_logger("test.module")

        assert isinstance(logger, ContextLogger)

    def test_with_data_creates_child_logger(self):
        """with_data should create a child logger with extra data."""
        logger = get_logger("test")
        child = logger.with_data(extra_key="extra_value")

        assert isinstance(child, ContextLogger)
        assert child.extra.get("extra_data", {}).get("extra_key") == "extra_value"


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    def test_duration_calculation(self):
        """PerformanceMetrics should calculate duration correctly."""
        import time

        metrics = PerformanceMetrics(operation="test_op", start_time=time.time())
        time.sleep(0.01)  # 10ms
        metrics.finish()

        assert metrics.duration_ms >= 10
        assert metrics.success is True

    def test_finish_with_error(self):
        """finish() should record error state."""
        metrics = PerformanceMetrics(operation="failing_op", start_time=0)
        metrics.finish(success=False, error="Test error")

        assert metrics.success is False
        assert metrics.error == "Test error"

    def test_to_dict(self):
        """to_dict should return all metrics."""
        metrics = PerformanceMetrics(
            operation="test_op",
            start_time=100,
            end_time=100.5,
            success=True,
            metadata={"key": "value"},
        )

        data = metrics.to_dict()

        assert data["operation"] == "test_op"
        assert data["duration_ms"] == 500.0
        assert data["success"] is True
        assert data["key"] == "value"


class TestPerformanceTimer:
    """Tests for PerformanceTimer context manager."""

    def test_context_manager_records_timing(self):
        """PerformanceTimer should record timing as context manager."""
        import time

        with PerformanceTimer("test_operation") as timer:
            time.sleep(0.01)  # 10ms
            timer.add_metadata("test_key", "test_value")

        assert timer.metrics.duration_ms >= 10
        assert timer.metrics.success is True
        assert timer.metrics.metadata["test_key"] == "test_value"

    def test_context_manager_captures_exception(self):
        """PerformanceTimer should capture exception details."""
        try:
            with PerformanceTimer("failing_operation") as timer:
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert timer.metrics.success is False
        assert "Test exception" in timer.metrics.error


class TestLogEvents:
    """Tests for LogEvents constants."""

    def test_all_events_are_strings(self):
        """All LogEvents should be strings."""
        events = [
            LogEvents.CLIENT_CONNECTED,
            LogEvents.CLIENT_DISCONNECTED,
            LogEvents.CLIENT_AUTHENTICATED,
            LogEvents.AUTH_FAILED,
            LogEvents.REQUEST_RECEIVED,
            LogEvents.REQUEST_QUEUED,
            LogEvents.REQUEST_STARTED,
            LogEvents.REQUEST_COMPLETED,
            LogEvents.REQUEST_FAILED,
            LogEvents.REQUEST_CANCELLED,
            LogEvents.REQUEST_TIMEOUT,
            LogEvents.MODEL_LOADING,
            LogEvents.MODEL_LOADED,
            LogEvents.SERVER_STARTED,
            LogEvents.SERVER_STOPPED,
            LogEvents.QUEUE_FULL,
            LogEvents.RATE_LIMITED,
        ]

        for event in events:
            assert isinstance(event, str)
            assert len(event) > 0

    def test_events_are_lowercase_with_underscores(self):
        """LogEvents should follow lowercase_underscore convention."""
        events = [
            LogEvents.CLIENT_CONNECTED,
            LogEvents.REQUEST_STARTED,
            LogEvents.SERVER_STOPPED,
        ]

        for event in events:
            assert event == event.lower(), f"{event} should be lowercase"
            assert " " not in event, f"{event} should not contain spaces"


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_logging_sets_level(self):
        """configure_logging should set the root logger level."""
        configure_logging(level="DEBUG", json_format=False, use_colors=False)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_logging_json_format(self):
        """configure_logging with json_format should use StructuredFormatter."""
        configure_logging(level="INFO", json_format=True)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        assert isinstance(handler.formatter, StructuredFormatter)

    def test_configure_logging_console_format(self):
        """configure_logging without json_format should use ConsoleFormatter."""
        configure_logging(level="INFO", json_format=False)

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        assert isinstance(handler.formatter, ConsoleFormatter)
