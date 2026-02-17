"""Lightweight metrics counters for forecast service observability.

These counters track fallback and degradation events. They are
designed to be scraped via structured logging. If prometheus_client
is added later, these can be backed by real Prometheus counters.
"""

import threading

import structlog

logger = structlog.get_logger(__name__)


class _Counter:
    """Thread-safe monotonic counter with structured-log emission."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._value += 1
            value = self._value
        logger.info(
            "metric.counter",
            metric=self.name,
            value=value,
            **(labels or {}),
        )

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset counter (for testing only)."""
        with self._lock:
            self._value = 0


# Model fallback: MLflow unavailable, fell back to pickle
model_fallback_total = _Counter(
    "forecast_model_fallback_total",
    "Times the forecaster fell back from MLflow to local pickle model.",
)

# Heuristic scoring: model not loaded or missing features, used heuristic
heuristic_fallback_total = _Counter(
    "forecast_heuristic_fallback_total",
    "Times the forecaster used heuristic scoring instead of the ML model.",
)

# Zero-mode fallback
zero_fallback_total = _Counter(
    "forecast_zero_fallback_total",
    "Times the forecaster returned zero probability due to fallback mode.",
)
