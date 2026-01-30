"""Inference event sink abstraction for pluggable logging backends.

This module provides a protocol (interface) for inference event sinks,
allowing the system to use different logging implementations (stdout,
file, PostgreSQL) without changing the inference evaluation code.

The default sink is determined by the INFERENCE_EVENT_SINK environment variable:
- "jsonl" (default): JSONL file sink (existing behavior)
- "stdout": Stdout sink (structured logging to console)
- "postgres": PostgreSQL-backed sink (requires database configuration)
- "none": No-op sink (disables inference logging)

Usage:
    from api.inference_event_sink import get_inference_event_sink

    sink = get_inference_event_sink()
    sink.log_event(event)
"""

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Protocol

from api.inference_log import InferenceEvent

logger = logging.getLogger(__name__)


class InferenceEventSink(Protocol):
    """Protocol defining the inference event sink interface.

    Any event sink must implement the log_event method. Query capabilities
    are optional and only implemented by sinks that support it.
    """

    def log_event(self, event: InferenceEvent) -> None:
        """Log an inference event.

        This method should be non-blocking and fail gracefully.
        Errors should be logged but not raised.

        Args:
            event: The inference event to log.
        """
        ...


def get_inference_event_sink_backend() -> str:
    """Get the configured inference event sink backend.

    Returns:
        The backend name from INFERENCE_EVENT_SINK env var, defaulting to "jsonl".
    """
    return os.getenv("INFERENCE_EVENT_SINK", "jsonl")


class NoOpSink:
    """No-operation sink that discards all events.

    Used when inference logging is disabled.
    """

    def log_event(self, event: InferenceEvent) -> None:
        """Discard the event (no-op)."""
        pass


class StdoutSink:
    """Sink that logs inference events to stdout as structured JSON.

    Useful for development, debugging, and integration with log aggregators.
    """

    def __init__(self):
        """Initialize stdout sink."""
        self._logger = logging.getLogger("inference.events")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def log_event(self, event: InferenceEvent) -> None:
        """Log the event to stdout as JSON."""
        try:
            event_dict = {
                "request_id": event.request_id,
                "timestamp": event.timestamp.isoformat(),
                "model_version": event.model_version,
                "rules_version": event.rules_version,
                "model_score": event.model_score,
                "final_score": event.final_score,
                "rule_impacts": [
                    {
                        "rule_id": ri.rule_id,
                        "is_shadow": ri.is_shadow,
                        "score_delta": ri.score_delta,
                        "details": ri.details,
                    }
                    for ri in event.rule_impacts
                ],
            }
            self._logger.info(f"InferenceEvent: {json.dumps(event_dict)}")
        except Exception as e:
            logger.warning(f"Failed to log inference event to stdout: {e}")


class JsonlFileSink:
    """Sink that logs inference events to a JSONL file.

    This is the existing default behavior, preserving backward compatibility.
    Thread-safe through locking.
    """

    def __init__(self, storage_path: str | Path | None = None):
        """Initialize JSONL file sink.

        Args:
            storage_path: Path to the JSONL file. If None, uses
                INFERENCE_LOG_PATH env var or defaults to
                data/inference_events.jsonl.
        """
        if storage_path is None:
            storage_path = os.getenv(
                "INFERENCE_LOG_PATH", "data/inference_events.jsonl"
            )
        self.storage_path = Path(storage_path)
        self._lock = Lock()

        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: InferenceEvent) -> None:
        """Log the event to JSONL file."""
        try:
            event_dict = {
                "request_id": event.request_id,
                "timestamp": event.timestamp.isoformat(),
                "model_version": event.model_version,
                "rules_version": event.rules_version,
                "model_score": event.model_score,
                "final_score": event.final_score,
                "rule_impacts": [
                    {
                        "rule_id": ri.rule_id,
                        "is_shadow": ri.is_shadow,
                        "score_delta": ri.score_delta,
                        "details": ri.details,
                    }
                    for ri in event.rule_impacts
                ],
            }

            with self._lock:
                with open(self.storage_path, "a") as f:
                    f.write(json.dumps(event_dict) + "\n")

        except Exception as e:
            logger.warning(f"Failed to log inference event to file: {e}")


# Global sink instance
_global_sink: InferenceEventSink | None = None
_sink_lock = Lock()


def get_inference_event_sink() -> InferenceEventSink:
    """Get the global inference event sink instance.

    The backend is selected based on INFERENCE_EVENT_SINK environment variable:
    - "jsonl" (default): JSONL file sink
    - "stdout": Stdout sink
    - "postgres": PostgreSQL sink
    - "none": No-op sink (disables logging)

    Returns:
        InferenceEventSink instance.
    """
    global _global_sink
    if _global_sink is None:
        with _sink_lock:
            if _global_sink is None:
                backend = get_inference_event_sink_backend()

                if backend == "none":
                    _global_sink = NoOpSink()
                    logger.info("Inference event logging disabled (no-op sink)")
                elif backend == "stdout":
                    _global_sink = StdoutSink()
                    logger.info("Using stdout inference event sink")
                elif backend == "postgres":
                    # Lazy import to avoid database dependencies in unit tests
                    try:
                        from api.postgres_inference_sink import PostgresInferenceSink

                        _global_sink = PostgresInferenceSink()
                        logger.info("Using PostgreSQL inference event sink")
                    except Exception as e:
                        logger.warning(
                            f"Failed to initialize PostgreSQL sink, "
                            f"falling back to JSONL: {e}"
                        )
                        _global_sink = JsonlFileSink()
                else:
                    # Default: JSONL file sink (existing behavior)
                    _global_sink = JsonlFileSink()
                    logger.info("Using JSONL file inference event sink")

    return _global_sink


def set_inference_event_sink(sink: InferenceEventSink) -> None:
    """Set the global inference event sink instance (for testing).

    Args:
        sink: InferenceEventSink instance to use.
    """
    global _global_sink
    with _sink_lock:
        _global_sink = sink


def reset_inference_event_sink() -> None:
    """Reset the global inference event sink to None (for testing)."""
    global _global_sink
    with _sink_lock:
        _global_sink = None
