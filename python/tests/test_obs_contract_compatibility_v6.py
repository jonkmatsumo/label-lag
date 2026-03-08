"""Compatibility guards for v6 diagnostics/drift operability contract refinements."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast.model_manager import ModelManager
from training.detect_drift import detect_drift


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


def test_v6_diagnostics_and_ml_health_contract_guard_fields_and_bounds():
    manager = _fresh_manager()
    diagnostics = manager.get_diagnostics()
    health = diagnostics["ml_health"]

    assert {"status", "warnings", "last_reload_status", "last_reload_ts"}.issubset(
        diagnostics.keys()
    )
    assert {
        "benchmark_last_run_ts",
        "feature_coverage_warning_last_seen_ts",
    }.issubset(diagnostics.keys())
    assert diagnostics["status"] in {"success", "failure", "unknown", "not_run"}
    assert diagnostics["last_reload_status"] in {"idle", "success", "failed"}
    assert diagnostics["benchmark_last_status"] in {
        None,
        "skipped_disabled",
        "skipped_sampled_out",
        "success",
        "failed",
        "unknown",
    }

    assert {
        "status",
        "warnings",
        "drift_resolution_mode",
        "drift_last_computed_ts",
    }.issubset(health.keys())
    assert health["status"] in {"success", "failure", "unknown", "not_run"}
    assert isinstance(health["warnings"], list)
    assert len(health["warnings"]) <= 4
    assert set(health["warnings"]).issubset(
        {
            "schema_mismatch_detected",
            "reload_failed_using_last_known_good",
            "feature_coverage_below_threshold",
            "drift_reference_unavailable",
        }
    )

    for ts_key in (
        "last_reload_ts",
        "benchmark_last_run_ts",
        "feature_coverage_warning_last_seen_ts",
    ):
        assert ts_key.endswith("_ts")
        assert diagnostics[ts_key] is None or isinstance(diagnostics[ts_key], float)

    for ts_key in (
        "last_reload_ts",
        "feature_coverage_last_seen_ts",
        "drift_last_computed_ts",
    ):
        assert ts_key.endswith("_ts")
        assert health[ts_key] is None or isinstance(health[ts_key], float)


@patch("training.detect_drift.get_reference_data")
@patch("training.detect_drift.get_live_data")
def test_v6_drift_contract_guard_reference_metadata_fields_and_bounds(
    mock_live, mock_ref
):
    stable_df = _stable_frame()
    mock_ref.return_value = (
        stable_df,
        {
            "requested_alias": "champion",
            "requested_mode": "alias",
            "resolution_strategy": "production_stage",
            "selected_model_version": "7",
            "selected_run_id": "run-v7",
            "resolution_warning": "alias_not_found_fallback",
        },
    )
    mock_live.return_value = stable_df.copy()

    result = detect_drift()

    assert {
        "timestamp",
        "resolution_mode",
        "reference_resolution_mode_requested",
        "reference_resolution_mode",
        "reference_model_version",
        "reference_model_version_chosen",
        "reference_alias_requested",
        "reference_resolution_warning",
        "reference_resolution",
    }.issubset(result.keys())

    assert isinstance(result["timestamp"], str)
    assert result["resolution_mode"] in {"alias", "stage", "latest", "none"}
    assert result["reference_resolution_mode"] == result["resolution_mode"]
    assert result["reference_resolution_mode_requested"] in {
        "alias",
        "stage",
        "latest",
        "none",
    }
    assert result["reference_model_version"] == result["reference_model_version_chosen"]
    assert result["reference_alias_requested"] == "champion"
    assert result["reference_resolution_warning"] in {
        None,
        "alias_not_found_fallback",
        "alias_ambiguous_selected_highest",
        "stage_fallback_used",
        "latest_fallback_used",
        "no_reference_versions_available",
    }

    assert result["reference_alias_requested"] is None or (
        len(result["reference_alias_requested"]) <= 64
    )
    assert result["reference_model_version_chosen"] is None or (
        len(result["reference_model_version_chosen"]) <= 64
    )
    assert result["reference_resolution_warning"] is None or (
        len(result["reference_resolution_warning"]) <= 64
    )
