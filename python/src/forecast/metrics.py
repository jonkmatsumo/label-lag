"""Lightweight metrics for forecast service observability.

These metrics track fallback/degradation events and model benchmark latency.
"""

from prometheus_client import Counter, Gauge, Histogram

# NOTE: When adding labels to these metrics, ensure they are low-cardinality.
# Avoid labels like 'request_id' or 'transaction_id' that could explode the
# number of time series in Prometheus.

# Model fallback: MLflow unavailable, fell back to pickle
model_fallback_total = Counter(
    "forecast_model_fallback_total",
    "Times the forecaster fell back from MLflow to local pickle model.",
    ["reason"],
)

# Model reload failure: both MLflow and fallback failed
model_reload_failure_total = Counter(
    "forecast_model_reload_failure_total",
    "Times the model reload failed (both MLflow and fallback).",
    ["reason"],
)

# Model schema mismatch: feature hash doesn't match
model_schema_mismatch_total = Counter(
    "forecast_model_schema_mismatch_total",
    "Times the loaded model feature schema hash did not match the computed hash.",
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

# Coverage SLO guardrail counter (bounded bucket label set only).
inference_feature_coverage_below_threshold_total = Counter(
    "inference_feature_coverage_below_threshold_total",
    "Times inference feature coverage fell below configured warning threshold.",
    ["bucket"],
)

# Model benchmark latency samples during load-time benchmark.
inference_benchmark_sample_latency_ms = Histogram(
    "forecast_inference_benchmark_sample_latency_ms",
    "Sample-level inference latency during model-load benchmark (milliseconds).",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000),
)

# Load-time benchmark percentile snapshots.
inference_benchmark_percentile_latency_ms = Gauge(
    "forecast_inference_benchmark_percentile_latency_ms",
    "Inference latency percentile snapshot from load-time benchmark (milliseconds).",
    ["percentile"],
)
