"""Compatibility guards for diagnostics + drift payload contracts (v8 tranche)."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from forecast.model_manager import ModelManager
from training.detect_drift import MIN_REFERENCE_SAMPLES, detect_drift
from training.reason_codes import (
    BENCHMARK_STATUSES,
    DIAGNOSTICS_DEGRADED_REASONS,
    DIAGNOSTICS_WARNING_CODES,
    DRIFT_ERROR_CODES,
    DRIFT_FALLBACK_REASONS,
    DRIFT_FEATURE_STATUSES,
    DRIFT_REFERENCE_RESOLUTION_WARNING_CODES,
    DRIFT_RESOLUTION_MODES,
    MODEL_MANAGER_STATES,
    OPERABILITY_STATUSES,
    RELOAD_FAILURE_REASONS,
    RELOAD_STATUSES,
)

DIAGNOSTICS_KEYS = {
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

ML_HEALTH_KEYS = {
    "model",
    "benchmark",
    "drift",
    "feature_coverage",
    "config",
    "warnings",
    "status",
    "overall_status",
    "degraded",
    "has_warnings",
    "warning_count",
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

DRIFT_KEYS = {
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

DRIFT_REFERENCE_KEYS = {
    "requested_alias",
    "resolution_strategy",
    "resolution_mode",
    "alias_candidate_count",
    "alias_ambiguous",
    "selected_model_version",
    "selected_run_id",
}

DRIFT_FEATURE_KEYS = {"psi", "status", "drift_error", "bucketing"}
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


def _stable_frame(size: int = 1000) -> pd.DataFrame:
    base = np.arange(size, dtype=float)
    return pd.DataFrame(
        {
            "velocity_24h": base,
            "amount_to_avg_ratio_30d": base * 0.5,
            "balance_volatility_z_score": base - 250.0,
        }
    )


def test_v8_diagnostics_contract_keys_and_summary_shape_are_stable():
    manager = _fresh_manager()
    manager._schema_mismatch_detected = True
    manager._model_source = "fallback"
    manager.update_feature_coverage_warning(active=True, observed_ts=111.5, ratio=0.4)
    mock_cache = SimpleNamespace(
        _cache=SimpleNamespace(
            computed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            result={
                "resolution_mode": "none",
                "error_code": "no_reference_data",
            },
        )
    )

    with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
        diagnostics = manager.get_diagnostics()

    assert set(diagnostics.keys()) == DIAGNOSTICS_KEYS
    assert diagnostics["state"] in MODEL_MANAGER_STATES
    assert diagnostics["status"] in OPERABILITY_STATUSES
    assert diagnostics["last_reload_status"] in RELOAD_STATUSES
    assert diagnostics["last_reload_reason"] in RELOAD_FAILURE_REASONS | {None}
    assert diagnostics["benchmark_last_status"] in BENCHMARK_STATUSES | {None}
    assert set(diagnostics["warnings"]).issubset(DIAGNOSTICS_WARNING_CODES)
    assert set(diagnostics["degraded_reasons"]).issubset(DIAGNOSTICS_DEGRADED_REASONS)

    health = diagnostics["ml_health"]
    assert set(health.keys()) == ML_HEALTH_KEYS
    assert health["overall_status"] == health["status"]
    assert health["status"] in OPERABILITY_STATUSES
    assert health["has_warnings"] == bool(health["warnings"])
    assert health["warning_count"] == len(health["warnings"])
    assert health["degraded"] == (
        bool(diagnostics["degraded_reasons"])
        or health["overall_status"] in {"failure", "unknown"}
    )


@patch("training.detect_drift.get_reference_data")
@patch("training.detect_drift.get_live_data")
def test_v8_drift_contract_shape_and_vocab_across_reference_error_modes(
    mock_live,
    mock_ref,
):
    stable = _stable_frame()
    sparse_size = MIN_REFERENCE_SAMPLES + 20
    sparse_reference = np.array([0.0] * (sparse_size - 10) + [1.0] * 10)
    sparse_live = np.array([0.0] * (sparse_size - 30) + [8.0] * 30)

    scenarios = [
        {
            "name": "success_alias",
            "reference": (
                stable,
                {
                    "requested_alias": "champion",
                    "requested_mode": "alias",
                    "resolution_strategy": "alias",
                    "selected_model_version": "9",
                    "selected_run_id": "run-v9",
                },
            ),
            "live": stable.copy(),
            "error_code": None,
            "resolution_mode": "alias",
            "resolution_warning": None,
        },
        {
            "name": "alias_fallback",
            "reference": (
                stable,
                {
                    "requested_alias": "champion",
                    "requested_mode": "alias",
                    "resolution_strategy": "production_stage",
                    "selected_model_version": "7",
                    "selected_run_id": "run-v7",
                    "resolution_warning": "alias_not_found_fallback",
                },
            ),
            "live": stable.copy(),
            "error_code": None,
            "resolution_mode": "stage",
            "resolution_warning": "alias_not_found_fallback",
        },
        {
            "name": "no_reference",
            "reference": (
                None,
                {
                    "requested_alias": "champion",
                    "requested_mode": "alias",
                    "resolution_warning": "no_reference_versions_available",
                },
            ),
            "live": pd.DataFrame(),
            "error_code": "no_reference_data",
            "resolution_mode": "none",
            "resolution_warning": "no_reference_versions_available",
        },
        {
            "name": "insufficient_reference",
            "reference": _stable_frame(size=50),
            "live": pd.DataFrame(),
            "error_code": "insufficient_reference_samples",
            "resolution_mode": "none",
            "resolution_warning": "no_reference_versions_available",
        },
        {
            "name": "suppressed_bucket_mass",
            "reference": pd.DataFrame(
                {
                    "velocity_24h": sparse_reference,
                    "amount_to_avg_ratio_30d": sparse_reference,
                    "balance_volatility_z_score": sparse_reference,
                }
            ),
            "live": pd.DataFrame(
                {
                    "velocity_24h": sparse_live,
                    "amount_to_avg_ratio_30d": sparse_live,
                    "balance_volatility_z_score": sparse_live,
                }
            ),
            "error_code": "insufficient_bucket_mass",
            "resolution_mode": "none",
            "resolution_warning": "no_reference_versions_available",
        },
    ]

    for scenario in scenarios:
        mock_ref.return_value = scenario["reference"]
        mock_live.return_value = scenario["live"]

        result = detect_drift()

        assert set(result.keys()) == DRIFT_KEYS, scenario["name"]
        assert isinstance(result["timestamp"], str), scenario["name"]
        assert result["resolution_mode"] in DRIFT_RESOLUTION_MODES, scenario["name"]
        assert result["reference_resolution_mode"] == result["resolution_mode"], (
            scenario["name"]
        )
        assert (
            result["reference_resolution_mode_requested"] in DRIFT_RESOLUTION_MODES
        ), scenario["name"]
        assert result["error_code"] == scenario["error_code"], scenario["name"]
        assert result["resolution_mode"] == scenario["resolution_mode"], scenario[
            "name"
        ]
        assert (
            result["reference_resolution_warning"] == scenario["resolution_warning"]
        ), scenario["name"]

        assert set(result["reference_resolution"].keys()) == DRIFT_REFERENCE_KEYS
        assert (
            result["reference_resolution"]["resolution_mode"] in DRIFT_RESOLUTION_MODES
        )
        assert (
            result["reference_resolution"]["resolution_strategy"]
            in DRIFT_RESOLUTION_MODES
        )

        assert result[
            "reference_resolution_warning"
        ] in DRIFT_REFERENCE_RESOLUTION_WARNING_CODES | {None}
        assert result["drift_error"] in DRIFT_ERROR_CODES | {None}
        if result["error_code"] is None:
            assert result["error_message"] is None
            assert result["error"] is None
            assert result["drift_error"] is None
        else:
            assert result["drift_error"] == result["error_code"]
            assert result["error_message"] is not None
            assert result["error"] == result["error_message"]

        for feature_result in result["features"].values():
            assert set(feature_result.keys()) == DRIFT_FEATURE_KEYS
            assert feature_result["status"] in DRIFT_FEATURE_STATUSES
            assert set(feature_result["bucketing"].keys()) == DRIFT_BUCKETING_KEYS
            assert feature_result["drift_error"] in DRIFT_ERROR_CODES | {None}

            bucketing = feature_result["bucketing"]
            assert bucketing["buckettype_requested"] in {"bins", "quantiles", None}
            assert bucketing["buckettype_used"] in {"bins", "quantiles", None}
            assert bucketing["bucketing_fallback_reason"] in (
                DRIFT_FALLBACK_REASONS | {None}
            )
            assert bucketing["drift_error"] in DRIFT_ERROR_CODES | {None}
            break
