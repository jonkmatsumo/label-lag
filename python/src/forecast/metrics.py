"""Lightweight metrics counters for forecast service observability.

These counters track fallback and degradation events.
"""

from prometheus_client import Counter, Histogram

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

# Feature coverage ratio distribution
inference_feature_coverage_ratio = Histogram(
    "inference_feature_coverage_ratio",
    "Ratio of required features present during inference (0.0 to 1.0).",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0),
)
