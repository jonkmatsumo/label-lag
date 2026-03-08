"""Compatibility guard tests for ML health and drift contract shapes."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast.model_manager import ModelManager
from training.detect_drift import (
    MAX_DRIFT_ERROR_MESSAGE_LENGTH,
    MIN_REFERENCE_SAMPLES,
    detect_drift,
)
from training.reason_codes import DriftErrorCode, DriftResolutionMode

ML_HEALTH_REQUIRED_KEYS = {
    "model",
    "benchmark",
    "drift",
    "feature_coverage",
    "config",
    "status",
    "state",
    "active_model_version",
    "last_reload_status",
    "schema_mismatch_detected",
    "feature_coverage_status",
    "drift_resolution_mode",
}
ML_HEALTH_OPTIONAL_KEYS = {
    "last_reload_ts",
    "benchmark_status",
    "feature_coverage_last_seen_ts",
    "drift_reference_available",
    "drift_last_computed_ts",
    "drift_last_error_code",
}
ML_HEALTH_MODEL_KEYS = {
    "state",
    "active_model_version",
    "last_reload_status",
    "last_reload_ts",
    "schema_mismatch_detected",
}
ML_HEALTH_BENCHMARK_KEYS = {"enabled", "last_status", "last_run_ts"}
ML_HEALTH_DRIFT_KEYS = {"reference_resolution_mode", "last_error_code"}
ML_HEALTH_FEATURE_COVERAGE_KEYS = {"last_ratio", "below_threshold"}
ML_HEALTH_CONFIG_KEYS = {
    "strict_feature_schema",
    "strict_tuning_resume_validation",
    "strict_split_strategy_validation",
}

DRIFT_REQUIRED_KEYS = {
    "timestamp",
    "hours_analyzed",
    "threshold",
    "reference_size",
    "live_size",
    "features",
    "drift_detected",
    "drifted_features",
    "drift_error",
    "error_code",
    "error_message",
    "resolution_mode",
    "alerts",
    "reference_resolution",
    "reference_model_version",
}
DRIFT_OPTIONAL_KEYS = {"error"}
DRIFT_REFERENCE_RESOLUTION_KEYS = {
    "requested_alias",
    "resolution_strategy",
    "resolution_mode",
    "alias_candidate_count",
    "alias_ambiguous",
    "selected_model_version",
    "selected_run_id",
}
DRIFT_BUCKETING_KEYS = {
    "buckettype_requested",
    "buckettype_used",
    "buckets_requested",
    "buckets_used",
    "bucketing_fallback_reason",
    "breakpoints",
    "reference_sample_size",
    "nonempty_buckets",
    "nonempty_buckets_ratio",
    "min_expected_count",
    "bucket_mass_ok",
    "bucket_mass_guardrail_applied",
    "drift_error",
}


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


def _assert_bounded_optional_str(value, *, max_len: int) -> None:
    assert value is None or (isinstance(value, str) and len(value) <= max_len)


def _assert_ml_health_contract(health: dict) -> None:
    assert set(health.keys()) == (ML_HEALTH_REQUIRED_KEYS | ML_HEALTH_OPTIONAL_KEYS)
    assert set(health["model"].keys()) == ML_HEALTH_MODEL_KEYS
    assert set(health["benchmark"].keys()) == ML_HEALTH_BENCHMARK_KEYS
    assert set(health["drift"].keys()) == ML_HEALTH_DRIFT_KEYS
    assert set(health["feature_coverage"].keys()) == ML_HEALTH_FEATURE_COVERAGE_KEYS
    assert set(health["config"].keys()) == ML_HEALTH_CONFIG_KEYS

    assert isinstance(health["state"], str)
    assert health["status"] in {"success", "failure", "unknown", "not_run"}
    assert isinstance(health["active_model_version"], str)
    assert isinstance(health["last_reload_status"], str)
    assert isinstance(health["schema_mismatch_detected"], bool)
    assert health["feature_coverage_status"] in {"ok", "warning"}
    assert health["drift_resolution_mode"] in {"alias", "stage", "latest", "none"}
    assert len(health["state"]) <= 32
    assert len(health["active_model_version"]) <= 64
    assert len(health["last_reload_status"]) <= 32
    assert len(health["drift_resolution_mode"]) <= 16
    _assert_bounded_optional_str(health["benchmark_status"], max_len=32)
    _assert_bounded_optional_str(health["drift_last_error_code"], max_len=64)

    assert health["last_reload_ts"] is None or isinstance(
        health["last_reload_ts"], float
    )
    assert health["feature_coverage_last_seen_ts"] is None or isinstance(
        health["feature_coverage_last_seen_ts"], float
    )
    assert health["drift_last_computed_ts"] is None or isinstance(
        health["drift_last_computed_ts"], float
    )
    assert health["drift_reference_available"] is None or isinstance(
        health["drift_reference_available"], bool
    )

    assert isinstance(health["benchmark"]["enabled"], bool)
    _assert_bounded_optional_str(health["benchmark"]["last_status"], max_len=32)
    assert health["benchmark"]["last_run_ts"] is None or isinstance(
        health["benchmark"]["last_run_ts"], float
    )
    assert health["feature_coverage"]["last_ratio"] is None or isinstance(
        health["feature_coverage"]["last_ratio"], float
    )
    assert isinstance(health["feature_coverage"]["below_threshold"], bool)
    assert health["drift"]["reference_resolution_mode"] in {
        "alias",
        "stage",
        "latest",
        "none",
    }
    _assert_bounded_optional_str(health["drift"]["last_error_code"], max_len=64)
    assert all(isinstance(value, bool) for value in health["config"].values())

    for key, value in health.items():
        if key in {"model", "benchmark", "drift", "feature_coverage", "config"}:
            continue
        assert not isinstance(value, dict | list | tuple | set)


def _assert_drift_contract(result: dict) -> None:
    keys = set(result.keys())
    assert DRIFT_REQUIRED_KEYS.issubset(keys)
    assert keys.issubset(DRIFT_REQUIRED_KEYS | DRIFT_OPTIONAL_KEYS)

    assert isinstance(result["timestamp"], str)
    assert isinstance(result["hours_analyzed"], int)
    assert isinstance(result["threshold"], int | float)
    assert isinstance(result["reference_size"], int)
    assert isinstance(result["live_size"], int)
    assert isinstance(result["features"], dict)
    assert isinstance(result["drift_detected"], bool)
    assert isinstance(result["drifted_features"], list)
    assert isinstance(result["alerts"], list)
    _assert_bounded_optional_str(result["drift_error"], max_len=64)
    _assert_bounded_optional_str(result["error_code"], max_len=64)
    _assert_bounded_optional_str(
        result["error_message"], max_len=MAX_DRIFT_ERROR_MESSAGE_LENGTH
    )
    _assert_bounded_optional_str(result["reference_model_version"], max_len=64)
    assert result["resolution_mode"] in {"alias", "stage", "latest", "none"}

    reference_resolution = result["reference_resolution"]
    assert isinstance(reference_resolution, dict)
    assert set(reference_resolution.keys()) == DRIFT_REFERENCE_RESOLUTION_KEYS
    assert reference_resolution["resolution_mode"] in {
        "alias",
        "stage",
        "latest",
        "none",
    }
    assert isinstance(reference_resolution["alias_candidate_count"], int)
    assert isinstance(reference_resolution["alias_ambiguous"], bool)
    _assert_bounded_optional_str(reference_resolution["requested_alias"], max_len=64)
    _assert_bounded_optional_str(
        reference_resolution["resolution_strategy"], max_len=64
    )
    _assert_bounded_optional_str(
        reference_resolution["selected_model_version"], max_len=64
    )
    _assert_bounded_optional_str(reference_resolution["selected_run_id"], max_len=128)
    if result["reference_model_version"] is not None:
        assert (
            result["reference_model_version"]
            == reference_resolution["selected_model_version"]
        )

    for feature_result in result["features"].values():
        assert "psi" in feature_result
        assert "status" in feature_result
        assert "bucketing" in feature_result
        assert feature_result["status"] in {"OK", "WARNING", "CRITICAL"}
        assert isinstance(feature_result["psi"], int | float)
        if "drift_error" in feature_result:
            _assert_bounded_optional_str(feature_result["drift_error"], max_len=64)

        bucketing = feature_result["bucketing"]
        assert isinstance(bucketing, dict)
        assert set(bucketing.keys()) == DRIFT_BUCKETING_KEYS
        assert isinstance(bucketing["breakpoints"], list)
        assert len(bucketing["breakpoints"]) <= 20
        _assert_bounded_optional_str(bucketing["drift_error"], max_len=64)
        for key, value in bucketing.items():
            if key == "breakpoints":
                continue
            assert not isinstance(value, dict | list | tuple | set)


def test_ml_health_contract_compatibility_shape_and_bounds():
    manager = _fresh_manager()
    manager.update_feature_coverage_warning(
        active=True, coverage_ratio=1.5, observed_ts=111.5
    )
    mock_cache = SimpleNamespace(
        _cache=SimpleNamespace(
            computed_at=datetime(2026, 1, 4, tzinfo=timezone.utc),
            result={
                "resolution_mode": "production_stage",
                "error_code": "x" * 100,
                "reference_resolution": {
                    "selected_run_id": "run-123",
                    "selected_model_version": "9",
                    "nested_unexpected_map": {"inner": "value"},
                },
            },
        )
    )

    with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
        health = manager.get_diagnostics()["ml_health"]

    _assert_ml_health_contract(health)


@patch("training.detect_drift.get_reference_data")
@patch("training.detect_drift.get_live_data")
def test_drift_contract_compatibility_success(mock_live, mock_ref):
    base = np.arange(1000, dtype=float)
    reference_df = pd.DataFrame(
        {
            "velocity_24h": base,
            "amount_to_avg_ratio_30d": base * 0.5,
            "balance_volatility_z_score": base - 250.0,
        }
    )
    mock_ref.return_value = (
        reference_df,
        {
            "resolution_strategy": "alias",
            "selected_model_version": "9",
            "selected_run_id": "run-v9",
        },
    )
    mock_live.return_value = reference_df.copy()

    result = detect_drift()

    _assert_drift_contract(result)
    assert result["error_code"] is None
    assert result["error_message"] is None
    assert result["resolution_mode"] == DriftResolutionMode.ALIAS.value


@patch("training.detect_drift.get_reference_data")
@patch("training.detect_drift.get_live_data")
def test_drift_contract_compatibility_no_reference(mock_live, mock_ref):
    mock_ref.return_value = None
    mock_live.return_value = pd.DataFrame()

    result = detect_drift()

    _assert_drift_contract(result)
    assert result["error_code"] == DriftErrorCode.NO_REFERENCE_DATA.value
    assert result["resolution_mode"] == DriftResolutionMode.NONE.value


@patch("training.detect_drift.get_reference_data")
@patch("training.detect_drift.get_live_data")
def test_drift_contract_compatibility_insufficient_reference_samples(
    mock_live, mock_ref
):
    mock_ref.return_value = pd.DataFrame(
        {
            "velocity_24h": [0.0] * 50,
            "amount_to_avg_ratio_30d": [0.0] * 50,
            "balance_volatility_z_score": [0.0] * 50,
        }
    )
    mock_live.return_value = pd.DataFrame()

    result = detect_drift()

    _assert_drift_contract(result)
    assert result["error_code"] == DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES.value
    assert result["resolution_mode"] == DriftResolutionMode.NONE.value


@patch("training.detect_drift.get_reference_data")
@patch("training.detect_drift.get_live_data")
def test_drift_contract_compatibility_suppressed_bucket_mass(mock_live, mock_ref):
    sample_size = MIN_REFERENCE_SAMPLES + 20
    sparse_reference = np.array([0.0] * (sample_size - 10) + [1.0] * 10)
    sparse_live = np.array([0.0] * (sample_size - 30) + [8.0] * 30)
    mock_ref.return_value = pd.DataFrame(
        {
            "velocity_24h": sparse_reference,
            "amount_to_avg_ratio_30d": sparse_reference,
            "balance_volatility_z_score": sparse_reference,
        }
    )
    mock_live.return_value = pd.DataFrame(
        {
            "velocity_24h": sparse_live,
            "amount_to_avg_ratio_30d": sparse_live,
            "balance_volatility_z_score": sparse_live,
        }
    )

    result = detect_drift()

    _assert_drift_contract(result)
    assert result["error_code"] == DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value
    for feature_result in result["features"].values():
        assert feature_result["bucketing"]["drift_error"] == (
            DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value
        )
