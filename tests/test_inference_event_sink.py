"""Tests for inference event sink abstraction."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from api.inference_event_sink import (
    JsonlFileSink,
    NoOpSink,
    StdoutSink,
    get_inference_event_sink,
    get_inference_event_sink_backend,
    reset_inference_event_sink,
    set_inference_event_sink,
)
from api.inference_log import InferenceEvent, RuleImpact


@pytest.fixture
def sample_event():
    """Create a sample inference event for testing."""
    return InferenceEvent(
        request_id="test-request-001",
        timestamp=datetime(2026, 1, 30, 12, 0, 0, tzinfo=timezone.utc),
        model_version="v1.0.0",
        rules_version="rules-v1",
        model_score=75,
        final_score=80,
        rule_impacts=[
            RuleImpact(
                rule_id="high_velocity",
                is_shadow=False,
                score_delta=5,
                details="Velocity exceeded threshold",
            ),
        ],
    )


@pytest.fixture(autouse=True)
def reset_sink():
    """Reset global sink before and after each test."""
    reset_inference_event_sink()
    yield
    reset_inference_event_sink()


class TestInferenceEventSinkBackendConfig:
    """Tests for inference event sink backend configuration."""

    def test_default_backend_is_jsonl(self, monkeypatch):
        """Test that default backend is jsonl when env var not set."""
        monkeypatch.delenv("INFERENCE_EVENT_SINK", raising=False)
        assert get_inference_event_sink_backend() == "jsonl"

    def test_backend_from_env_var(self, monkeypatch):
        """Test that backend is read from INFERENCE_EVENT_SINK env var."""
        monkeypatch.setenv("INFERENCE_EVENT_SINK", "stdout")
        assert get_inference_event_sink_backend() == "stdout"

    def test_postgres_backend_value(self, monkeypatch):
        """Test that postgres backend value is recognized."""
        monkeypatch.setenv("INFERENCE_EVENT_SINK", "postgres")
        assert get_inference_event_sink_backend() == "postgres"

    def test_none_backend_value(self, monkeypatch):
        """Test that none backend value is recognized."""
        monkeypatch.setenv("INFERENCE_EVENT_SINK", "none")
        assert get_inference_event_sink_backend() == "none"


class TestNoOpSink:
    """Tests for NoOpSink."""

    def test_log_event_does_nothing(self, sample_event):
        """Test that NoOpSink silently discards events."""
        sink = NoOpSink()
        # Should not raise
        sink.log_event(sample_event)

    def test_get_sink_returns_noop_when_disabled(self, monkeypatch):
        """Test that INFERENCE_EVENT_SINK=none returns NoOpSink."""
        monkeypatch.setenv("INFERENCE_EVENT_SINK", "none")
        sink = get_inference_event_sink()
        assert isinstance(sink, NoOpSink)


class TestStdoutSink:
    """Tests for StdoutSink."""

    def test_log_event_does_not_raise(self, sample_event):
        """Test that StdoutSink.log_event doesn't raise."""
        sink = StdoutSink()
        # Should not raise
        sink.log_event(sample_event)

    def test_get_sink_returns_stdout_when_configured(self, monkeypatch):
        """Test that INFERENCE_EVENT_SINK=stdout returns StdoutSink."""
        monkeypatch.setenv("INFERENCE_EVENT_SINK", "stdout")
        sink = get_inference_event_sink()
        assert isinstance(sink, StdoutSink)


class TestJsonlFileSink:
    """Tests for JsonlFileSink."""

    def test_log_event_writes_to_file(self, sample_event, tmp_path):
        """Test that JsonlFileSink writes events to file."""
        log_path = tmp_path / "events.jsonl"
        sink = JsonlFileSink(storage_path=log_path)

        sink.log_event(sample_event)

        # Check file was created and has content
        assert log_path.exists()
        content = log_path.read_text()
        assert "test-request-001" in content
        assert "high_velocity" in content

    def test_multiple_events_append(self, sample_event, tmp_path):
        """Test that multiple events are appended to the file."""
        log_path = tmp_path / "events.jsonl"
        sink = JsonlFileSink(storage_path=log_path)

        # Log multiple events
        sink.log_event(sample_event)
        sink.log_event(sample_event)
        sink.log_event(sample_event)

        # Check file has 3 lines
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_get_sink_returns_jsonl_by_default(self, monkeypatch):
        """Test that default sink is JsonlFileSink."""
        monkeypatch.delenv("INFERENCE_EVENT_SINK", raising=False)
        sink = get_inference_event_sink()
        assert isinstance(sink, JsonlFileSink)

    def test_log_event_handles_errors_gracefully(self, sample_event, tmp_path):
        """Test that file write errors are logged but not raised."""
        log_path = tmp_path / "events.jsonl"
        sink = JsonlFileSink(storage_path=log_path)

        # Make the file read-only after sink creation
        log_path.touch()
        log_path.chmod(0o444)

        try:
            # Should not raise even if write fails
            sink.log_event(sample_event)
        finally:
            # Restore permissions for cleanup
            log_path.chmod(0o644)


class TestSetInferenceEventSink:
    """Tests for set_inference_event_sink."""

    def test_set_sink_overrides_default(self, sample_event):
        """Test that set_inference_event_sink overrides the default."""
        mock_sink = MagicMock()
        set_inference_event_sink(mock_sink)

        sink = get_inference_event_sink()
        sink.log_event(sample_event)

        mock_sink.log_event.assert_called_once_with(sample_event)


class TestSinkGracefulDegradation:
    """Tests for graceful degradation behavior."""

    def test_postgres_fallback_on_error(self, monkeypatch):
        """Test that postgres failure falls back to JSONL."""
        monkeypatch.setenv("INFERENCE_EVENT_SINK", "postgres")

        # Mock the postgres import to fail
        with patch.dict(
            "sys.modules",
            {"api.postgres_inference_sink": None},
        ):
            # Force re-initialization
            reset_inference_event_sink()
            sink = get_inference_event_sink()

            # Should fall back to JSONL
            assert isinstance(sink, JsonlFileSink)
