"""Contract-shape golden tests for ML operability payloads."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast.model_manager import ModelManager
from training.detect_drift import (
    MAX_DRIFT_ERROR_MESSAGE_LENGTH,
    MONITORED_FEATURES,
    detect_drift,
)
from training.reason_codes import DriftErrorCode, DriftResolutionMode


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


def test_ml_health_contract_shape_and_bounds():
    manager = _fresh_manager()
    manager.update_feature_coverage_warning(
        active=True,
        coverage_ratio=3.5,
        observed_ts=111.5,
    )

    mock_cache = SimpleNamespace(
        _cache=SimpleNamespace(
            computed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            result={
                "reference_resolution": {
                    "resolution_strategy": "alias",
                    "selected_run_id": "run-123",
                },
                "error_code": "no_reference_data",
            },
        )
    )

    with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
        health = manager.get_ml_health_summary()

    assert set(health.keys()) == {
        "model",
        "benchmark",
        "drift",
        "feature_coverage",
        "config",
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
    assert set(health["config"].keys()) == {
        "strict_feature_schema",
        "strict_tuning_resume_validation",
        "strict_split_strategy_validation",
    }
    assert all(isinstance(value, bool) for value in health["config"].values())
    assert set(health["model"].keys()) == {
        "state",
        "active_model_version",
        "last_reload_status",
        "last_reload_ts",
        "schema_mismatch_detected",
    }
    assert set(health["benchmark"].keys()) == {"enabled", "last_status", "last_run_ts"}
    assert set(health["drift"].keys()) == {
        "reference_resolution_mode",
        "last_error_code",
    }
    assert set(health["feature_coverage"].keys()) == {"last_ratio", "below_threshold"}
    assert health["status"] in {"success", "failure", "unknown", "not_run"}
    assert isinstance(health["model"]["state"], str)
    assert isinstance(health["model"]["active_model_version"], str)
    assert isinstance(health["model"]["last_reload_status"], str)
    assert health["model"]["last_reload_ts"] is None or isinstance(
        health["model"]["last_reload_ts"], float
    )
    assert isinstance(health["model"]["schema_mismatch_detected"], bool)
    assert isinstance(health["benchmark"]["enabled"], bool)
    assert health["benchmark"]["last_status"] is None or isinstance(
        health["benchmark"]["last_status"], str
    )
    assert health["benchmark"]["last_run_ts"] is None or isinstance(
        health["benchmark"]["last_run_ts"], float
    )
    assert health["drift"]["reference_resolution_mode"] in {
        "alias",
        "stage",
        "latest",
        "none",
    }
    assert health["drift"]["last_error_code"] is None or isinstance(
        health["drift"]["last_error_code"], str
    )
    assert health["feature_coverage"]["last_ratio"] is None or isinstance(
        health["feature_coverage"]["last_ratio"], float
    )
    assert isinstance(health["feature_coverage"]["below_threshold"], bool)
    assert health["feature_coverage"]["last_ratio"] == 1.0
    assert 0.0 <= health["feature_coverage"]["last_ratio"] <= 1.0
    assert len(health["active_model_version"]) <= 64
    assert len(health["last_reload_status"]) <= 32
    assert (
        health["drift_last_error_code"] is None
        or len(health["drift_last_error_code"]) <= 64
    )
    for value in health.values():
        assert not isinstance(value, list | tuple | set)


@patch("training.detect_drift.get_reference_data")
@patch("training.detect_drift.get_live_data")
def test_drift_contract_shape_and_bounds(mock_live, mock_ref):
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

    assert set(result.keys()) == {
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
    }
    assert result["resolution_mode"] in {"alias", "stage", "latest", "none"}
    assert result["reference_model_version"] == "9"
    assert len(result["features"]) <= len(MONITORED_FEATURES)
    assert len(result["drifted_features"]) <= len(MONITORED_FEATURES)
    assert len(result["alerts"]) <= len(MONITORED_FEATURES)
    assert (
        result["error_message"] is None
        or len(result["error_message"]) <= MAX_DRIFT_ERROR_MESSAGE_LENGTH
    )
    assert result["error"] is None

    for feature_name, feature_result in result["features"].items():
        assert feature_name in MONITORED_FEATURES
        assert set(feature_result.keys()) == {
            "psi",
            "status",
            "drift_error",
            "bucketing",
        }
        assert feature_result["status"] in {"OK", "WARNING", "CRITICAL"}
        assert feature_result["drift_error"] is None or isinstance(
            feature_result["drift_error"], str
        )
        breakpoints = feature_result["bucketing"].get("breakpoints")
        if isinstance(breakpoints, list):
            assert len(breakpoints) <= 20


@patch("training.detect_drift.get_reference_data")
@patch("training.detect_drift.get_live_data")
def test_drift_error_contract_shape_and_bounds(mock_live, mock_ref):
    mock_ref.return_value = None
    mock_live.return_value = pd.DataFrame()

    result = detect_drift()

    assert set(result.keys()) == {
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
        "error",
    }
    assert result["error_code"] == DriftErrorCode.NO_REFERENCE_DATA.value
    assert len(result["error_code"]) <= 64
    assert isinstance(result["error_message"], str)
    assert result["error_message"] == "No reference data available"
    assert len(result["error_message"]) <= MAX_DRIFT_ERROR_MESSAGE_LENGTH
    assert result["error"] == result["error_message"]
    assert result["resolution_mode"] == DriftResolutionMode.NONE.value
