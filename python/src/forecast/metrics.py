"""Lightweight metrics counters for forecast service observability.

These counters track fallback and degradation events.
"""

from prometheus_client import Counter

# Model fallback: MLflow unavailable, fell back to pickle
model_fallback_total = Counter(
    "forecast_model_fallback_total",
    "Times the forecaster fell back from MLflow to local pickle model.",
    ["reason"],
)

# Heuristic scoring: model not loaded or missing features, used heuristic
heuristic_fallback_total = Counter(
    "forecast_heuristic_fallback_total",
    "Times the forecaster used heuristic scoring instead of the ML model.",
    ["reason"],
)

# Zero-mode fallback
zero_fallback_total = Counter(
    "forecast_zero_fallback_total",
    "Times the forecaster returned zero probability due to fallback mode.",
    ["reason"],
)
