"""Compatibility guardrails for ML health + drift contract payloads."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from forecast.model_manager import ModelManager
from training.detect_drift import (
    MAX_DRIFT_ERROR_MESSAGE_LENGTH,
    MIN_REFERENCE_SAMPLES,
    detect_drift,
)
from training.reason_codes import DriftErrorCode, DriftResolutionMode

ML_HEALTH_KEYS = {
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

DRIFT_RESULT_KEYS = {
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
    "error",
    "resolution_mode",
    "alerts",
    "reference_resolution",
    "reference_model_version",
    "reference_resolution_mode_requested",
    "reference_resolution_mode",
    "reference_model_version_chosen",
    "reference_alias_requested",
    "reference_resolution_warning",
}
REFERENCE_RESOLUTION_KEYS = {
    "requested_alias",
    "resolution_strategy",
    "resolution_mode",
    "alias_candidate_count",
    "alias_ambiguous",
    "selected_model_version",
    "selected_run_id",
}
DRIFT_FEATURE_KEYS = {"psi", "status", "drift_error", "bucketing"}
BUCKETING_KEYS = {
    "buckettype_requested",
    "buckettype_used",
    "buckets_requested",
    "buckets_used",
    "bucketing_fallback_reason",
    "reference_sample_size",
    "nonempty_buckets",
    "nonempty_buckets_ratio",
    "min_expected_count",
    "bucket_mass_ok",
    "bucket_mass_guardrail_applied",
    "drift_error",
    "breakpoints",
}


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


def _stable_frame(size: int = 1000) -> pd.DataFrame:
    base = np.arange(size, dtype=float)
    return pd.DataFrame(
        {
            "velocity_24h": base,
            "amount_to_avg_ratio_30d": base * 0.5,
            "balance_volatility_z_score": base - 250.0,
        }
    )


def test_ml_health_contract_guard_shape_types_and_bounds():
    manager = _fresh_manager()
    manager.update_feature_coverage_warning(
        active=True, coverage_ratio=0.42, observed_ts=100.5
    )
    mock_cache = SimpleNamespace(
        _cache=SimpleNamespace(
            computed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            result={
                "resolution_mode": "stage",
                "error_code": "no_reference_data",
                "reference_resolution": {"selected_run_id": "run-123"},
            },
        )
    )

    with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
        health = manager.get_diagnostics()["ml_health"]

    assert set(health.keys()) == ML_HEALTH_KEYS
    assert set(health["model"].keys()) == ML_HEALTH_MODEL_KEYS
    assert set(health["benchmark"].keys()) == ML_HEALTH_BENCHMARK_KEYS
    assert set(health["drift"].keys()) == ML_HEALTH_DRIFT_KEYS
    assert set(health["feature_coverage"].keys()) == ML_HEALTH_FEATURE_COVERAGE_KEYS
    assert set(health["config"].keys()) == ML_HEALTH_CONFIG_KEYS

    assert isinstance(health["state"], str)
    assert isinstance(health["warnings"], list)
    assert set(health["warnings"]).issubset(
        {
            "schema_mismatch_detected",
            "reload_failed_using_last_known_good",
            "feature_coverage_below_threshold",
            "drift_reference_unavailable",
        }
    )
    assert health["status"] in {"success", "failure", "unknown", "not_run"}
    assert isinstance(health["active_model_version"], str)
    assert isinstance(health["last_reload_status"], str)
    assert len(health["active_model_version"]) <= 64
    assert len(health["last_reload_status"]) <= 32
    assert health["last_reload_ts"] is None or isinstance(
        health["last_reload_ts"], float
    )
    assert isinstance(health["schema_mismatch_detected"], bool)
    assert isinstance(health["benchmark"]["enabled"], bool)
    assert health["benchmark"]["last_status"] is None or isinstance(
        health["benchmark"]["last_status"], str
    )
    assert health["benchmark"]["last_run_ts"] is None or isinstance(
        health["benchmark"]["last_run_ts"], float
    )
    assert health["drift"]["last_error_code"] is None or isinstance(
        health["drift"]["last_error_code"], str
    )
    assert health["feature_coverage"]["last_ratio"] is None or isinstance(
        health["feature_coverage"]["last_ratio"], float
    )
    assert isinstance(health["feature_coverage"]["below_threshold"], bool)
    assert all(isinstance(value, bool) for value in health["config"].values())
    assert health["drift_resolution_mode"] in {"alias", "stage", "latest", "none"}
    assert health["drift"]["reference_resolution_mode"] in {
        "alias",
        "stage",
        "latest",
        "none",
    }
    assert (
        health["drift_last_error_code"] is None
        or len(health["drift_last_error_code"]) <= 64
    )
    assert all(
        not isinstance(value, list | tuple | set)
        for key, value in health.items()
        if key
        not in {"model", "benchmark", "drift", "feature_coverage", "config", "warnings"}
    )


@pytest.mark.parametrize(
    ("scenario", "expected_code", "expected_mode", "expected_version"),
    [
        (
            "success",
            None,
            DriftResolutionMode.ALIAS.value,
            "9",
        ),
        (
            "no_reference",
            DriftErrorCode.NO_REFERENCE_DATA.value,
            DriftResolutionMode.NONE.value,
            None,
        ),
        (
            "insufficient_reference",
            DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES.value,
            DriftResolutionMode.NONE.value,
            None,
        ),
        (
            "suppressed_bucket_mass",
            DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value,
            DriftResolutionMode.NONE.value,
            None,
        ),
    ],
)
@patch("training.detect_drift.get_reference_data")
@patch("training.detect_drift.get_live_data")
def test_drift_contract_guard_shape_across_modes(
    mock_live,
    mock_ref,
    scenario,
    expected_code,
    expected_mode,
    expected_version,
):
    stable_df = _stable_frame()
    if scenario == "success":
        mock_ref.return_value = (
            stable_df,
            {
                "resolution_strategy": "alias",
                "selected_model_version": "9",
                "selected_run_id": "run-v9",
            },
        )
        mock_live.return_value = stable_df.copy()
    elif scenario == "no_reference":
        mock_ref.return_value = None
        mock_live.return_value = pd.DataFrame()
    elif scenario == "insufficient_reference":
        mock_ref.return_value = _stable_frame(size=50)
        mock_live.return_value = pd.DataFrame()
    elif scenario == "suppressed_bucket_mass":
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
    else:
        raise AssertionError(f"Unsupported scenario: {scenario}")

    result = detect_drift()

    assert set(result.keys()) == DRIFT_RESULT_KEYS
    assert result["error_code"] == expected_code
    assert result["resolution_mode"] == expected_mode
    assert result["reference_model_version"] == expected_version
    assert result["reference_model_version_chosen"] == expected_version
    assert isinstance(result["timestamp"], str)
    assert (
        result["error_message"] is None
        or len(result["error_message"]) <= MAX_DRIFT_ERROR_MESSAGE_LENGTH
    )
    assert set(result["reference_resolution"].keys()) == REFERENCE_RESOLUTION_KEYS
    assert result["reference_resolution_mode"] == result["resolution_mode"]
    assert result["reference_resolution_mode_requested"] in {
        DriftResolutionMode.ALIAS.value,
        DriftResolutionMode.STAGE.value,
        DriftResolutionMode.LATEST.value,
        DriftResolutionMode.NONE.value,
    }
    assert result["reference_alias_requested"] is None or isinstance(
        result["reference_alias_requested"], str
    )
    assert result["reference_resolution_warning"] is None or isinstance(
        result["reference_resolution_warning"], str
    )
    assert (
        result["reference_resolution"]["resolution_mode"] == result["resolution_mode"]
    )
    assert (
        result["reference_resolution"]["resolution_strategy"]
        == result["resolution_mode"]
    )
    assert isinstance(result["reference_resolution"]["alias_candidate_count"], int)
    assert isinstance(result["reference_resolution"]["alias_ambiguous"], bool)
    assert (
        result["reference_resolution"]["selected_model_version"] is None
        or len(result["reference_resolution"]["selected_model_version"]) <= 64
    )
    assert (
        result["reference_resolution"]["selected_run_id"] is None
        or len(result["reference_resolution"]["selected_run_id"]) <= 128
    )
    if result["error_code"] is None:
        assert result["error"] is None
    else:
        assert result["error"] == result["error_message"]

    for feature_result in result["features"].values():
        assert set(feature_result.keys()) == DRIFT_FEATURE_KEYS
        assert set(feature_result["bucketing"].keys()) == BUCKETING_KEYS
        assert feature_result["status"] in {"OK", "WARNING", "CRITICAL"}
        assert len(feature_result["status"]) <= 16
        assert (
            feature_result["drift_error"] is None
            or len(feature_result["drift_error"]) <= 64
        )
        breakpoints = feature_result["bucketing"]["breakpoints"]
        if isinstance(breakpoints, list):
            assert len(breakpoints) <= 20
