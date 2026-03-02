"""Golden-shape guardrails for ML health and drift payloads."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast.model_manager import ModelManager
from training.detect_drift import MONITORED_FEATURES, detect_drift


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


def test_ml_health_payload_golden_shape_and_bounds():
    manager = _fresh_manager()
    manager.update_feature_coverage_warning(active=True, observed_ts=111.5)

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
        health = manager.get_diagnostics()["ml_health"]

    assert set(health.keys()) == {
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
        "config",
    }
    assert health["state"] in {"idle", "loading", "ready", "failed"}
    assert len(health["active_model_version"]) <= 64
    assert len(health["last_reload_status"]) <= 32
    assert health["drift_resolution_mode"] in {"alias", "stage", "latest", "none"}
    assert (
        health["drift_last_error_code"] is None
        or len(health["drift_last_error_code"]) <= 64
    )
    assert set(health["config"].keys()) == {
        "strict_feature_schema",
        "strict_tuning_resume_validation",
        "strict_split_strategy_validation",
    }
    assert all(isinstance(value, bool) for value in health["config"].values())


@patch("training.detect_drift.get_reference_data")
@patch("training.detect_drift.get_live_data")
def test_detect_drift_payload_golden_shape_and_bounds(mock_live, mock_ref):
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
        "resolution_mode",
        "alerts",
        "reference_resolution",
    }
    assert result["resolution_mode"] in {"alias", "stage", "latest", "none"}
    assert isinstance(result["features"], dict)
    assert len(result["features"]) <= len(MONITORED_FEATURES)
    assert len(result["drifted_features"]) <= len(MONITORED_FEATURES)
    assert len(result["alerts"]) <= len(MONITORED_FEATURES)
    assert result["error_message"] is None or len(result["error_message"]) <= 120

    for feature_name, feature_result in result["features"].items():
        assert feature_name in MONITORED_FEATURES
        assert set(feature_result.keys()) == {"psi", "status", "bucketing"}
        assert feature_result["status"] in {"OK", "WARNING", "CRITICAL"}
        assert len(feature_result["status"]) <= 16
