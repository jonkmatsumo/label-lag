"""Inference event sink for asynchronous event logging."""

import logging
import os
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any

from api.inference_log import InferenceEvent

logger = logging.getLogger(__name__)


class InferenceEventSink(ABC):
    """Abstract base class for inference event sinks."""

    @abstractmethod
    def log_event(self, event: InferenceEvent) -> None:
        """Log an inference event.

        Args:
            event: The event to log.
        """
        pass


class NoOpSink(InferenceEventSink):
    """Event sink that does nothing."""

    def log_event(self, event: InferenceEvent) -> None:
        pass


class StdoutSink(InferenceEventSink):
    """Event sink that logs to stdout (via logger)."""

    def log_event(self, event: InferenceEvent) -> None:
        logger.info(f"Inference event: {event.to_dict()}")


class JsonlFileSink(InferenceEventSink):
    """Event sink that logs to a JSONL file."""

    def __init__(self, log_path: str = "data/inference_events.jsonl"):
        from api.inference_log import InferenceLogger
        self.logger = InferenceLogger(log_path=log_path)

    def log_event(self, event: InferenceEvent) -> None:
        self.logger.log_event(event)


def get_inference_event_sink_backend() -> str:
    """Get the configured inference event sink backend.

    Returns:
        The backend name from INFERENCE_EVENT_SINK environment variable.
        Defaults to "none" for compute-only operation.
    """
    return os.getenv("INFERENCE_EVENT_SINK", "none")


_global_sink: InferenceEventSink | None = None
_sink_lock = Lock()


def get_inference_event_sink() -> InferenceEventSink:
    """Get the global inference event sink instance.

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
                elif backend == "jsonl":
                    _global_sink = JsonlFileSink()
                    logger.info("Using JSONL file inference event sink")
                elif backend == "postgres":
                    from api.grpc_inference_sink import GrpcInferenceSink

                    _global_sink = GrpcInferenceSink()
                    logger.info("Using gRPC inference sink (via Analytics service)")
                else:
                    _global_sink = NoOpSink()
                    logger.warning(
                        f"Unknown inference event sink backend: {backend}. "
                        "Defaulting to no-op."
                    )

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