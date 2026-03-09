"""Canonicalization guards for diagnostics + ml_health payload fields."""

from types import SimpleNamespace
from unittest.mock import patch

from forecast.model_manager import ModelManager
from training.reason_codes import (
    BENCHMARK_STATUSES,
    MODEL_MANAGER_STATES,
    OPERABILITY_STATUSES,
    RELOAD_FAILURE_REASONS,
    RELOAD_STATUSES,
)

_DIAGNOSTICS_KEYS = {
    "status",
    "state",
    "model_version",
    "model_source",
    "last_error",
    "schema_mismatch_detected",
    "calibrator_loaded",
    "has_bundle",
    "last_reload_ts",
    "last_reload_status",
    "last_reload_reason",
    "benchmark_last_run_ts",
    "benchmark_last_status",
    "degraded_reasons",
    "active_model_version",
    "feature_coverage_warning_active",
    "feature_coverage_last_ratio",
    "feature_coverage_warning_last_seen_ts",
    "ml.training.run_id",
    "ml.model.version",
    "ml.feature.schema_hash",
    "config",
    "warnings",
    "ml_health",
}

_ML_HEALTH_KEYS = {
    "model",
    "benchmark",
    "drift",
    "feature_coverage",
    "config",
    "warnings",
    "status",
    "state",
    "active_model_version",
    "last_reload_status",
    "last_reload_ts",
    "schema_mismatch_detected",
    "benchmark_status",
    "feature_coverage_status",
    "feature_coverage_last_seen_ts",
    "drift_reference_available",
    "drift_resolution_mode",
    "drift_last_computed_ts",
    "drift_last_error_code",
}


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


def test_diagnostics_payload_keys_types_and_bounds_are_canonical():
    manager = _fresh_manager()
    manager._state = "INVALID-STATE-" * 20
    manager._model_version = "v" * 300
    manager._model_source = "external-source-name" * 20
    manager._last_error = "error-" * 80
    manager._mlflow_failure_reason = "unknown-failure-with-extra-context" * 10
    manager._feature_coverage_warning_active = "yes"
    manager._feature_coverage_last_ratio = "1.7"
    manager._feature_coverage_warning_last_seen_ts = "not-a-float"
    manager._benchmark_last_status = "benchmark failed for detailed operator reason"
    manager._training_identity = {
        "mlflow_run_id": "r" * 300,
        "model_version": "m" * 300,
        "feature_schema_hash": "h" * 300,
    }

    diagnostics = manager.get_diagnostics()

    assert set(diagnostics.keys()) == _DIAGNOSTICS_KEYS
    assert diagnostics["state"] in MODEL_MANAGER_STATES
    assert diagnostics["status"] in OPERABILITY_STATUSES
    assert diagnostics["last_reload_status"] in RELOAD_STATUSES
    assert diagnostics["benchmark_last_status"] in BENCHMARK_STATUSES | {None}
    assert diagnostics["model_source"] in {"mlflow", "fallback", "none"}
    assert diagnostics["last_reload_reason"] in RELOAD_FAILURE_REASONS | {None}

    assert isinstance(diagnostics["warnings"], list)
    assert isinstance(diagnostics["degraded_reasons"], list)
    assert isinstance(diagnostics["schema_mismatch_detected"], bool)
    assert isinstance(diagnostics["feature_coverage_warning_active"], bool)
    assert diagnostics["feature_coverage_warning_active"] is True

    assert isinstance(diagnostics["model_version"], str)
    assert isinstance(diagnostics["active_model_version"], str)
    assert len(diagnostics["model_version"]) <= 64
    assert len(diagnostics["active_model_version"]) <= 64
    assert diagnostics["feature_coverage_last_ratio"] == 1.0
    assert diagnostics["feature_coverage_warning_last_seen_ts"] is None
    assert diagnostics["last_error"] is not None
    assert len(diagnostics["last_error"]) <= 200
    assert len(diagnostics["ml.training.run_id"]) <= 128
    assert len(diagnostics["ml.model.version"]) <= 64
    assert len(diagnostics["ml.feature.schema_hash"]) <= 128


def test_ml_health_summary_canonicalizes_optional_fields_and_scalar_types():
    manager = _fresh_manager()
    snapshot = {
        "state": "unexpected_state_value",
        "status": "not-a-real-status",
        "active_model_version": "v" * 400,
        "last_reload_status": "reload-status-" * 30,
        "last_reload_ts": "nan",
        "schema_mismatch_detected": "yes",
        "benchmark_last_status": "not-a-status",
        "benchmark_last_run_ts": "not-a-float",
        "feature_coverage_warning_active": "true",
        "feature_coverage_last_ratio": -2.0,
        "feature_coverage_warning_last_seen_ts": "not-a-float",
        "warnings": [
            "feature_coverage_below_threshold",
            "ad-hoc operator warning text",
            "",
        ],
        "config": {
            "strict_feature_schema": "1",
            "strict_tuning_resume_validation": "0",
            "strict_split_strategy_validation": 1,
        },
    }

    with patch(
        "forecast.drift_cache.get_drift_cache",
        return_value=SimpleNamespace(_cache=None),
    ):
        health = manager._build_ml_health_summary(snapshot)

    assert set(health.keys()) == _ML_HEALTH_KEYS
    assert health["state"] in MODEL_MANAGER_STATES
    assert health["status"] in OPERABILITY_STATUSES
    assert len(health["active_model_version"]) <= 64
    assert len(health["last_reload_status"]) <= 32
    assert health["benchmark"]["last_status"] == "unknown"
    assert health["benchmark"]["last_run_ts"] is None
    assert health["last_reload_ts"] is None
    assert health["feature_coverage_last_seen_ts"] is None
    assert health["feature_coverage"]["last_ratio"] == 0.0
    assert health["feature_coverage"]["below_threshold"] is True
    assert health["warnings"] == ["feature_coverage_below_threshold"]
