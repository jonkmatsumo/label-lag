"""Guardrail tests for ModelManager default/warn-only behavior."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from forecast.model_manager import ModelManager
from training.reason_codes import (
    MODEL_MANAGER_BASELINE_DIAGNOSTIC_KEYS,
    OPERABILITY_STATUSES,
)


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


def test_diagnostics_snapshot_includes_required_baseline_fields():
    manager = _fresh_manager()
    diag = manager.get_diagnostics()

    expected_keys = {
        "state",
        "active_model_version",
        "last_reload_status",
        "schema_mismatch_detected",
    }
    assert expected_keys.issubset(diag.keys())
    assert MODEL_MANAGER_BASELINE_DIAGNOSTIC_KEYS.issubset(diag.keys())


def test_missing_registry_features_warn_only_when_strict_flag_unset(monkeypatch):
    monkeypatch.delenv("ENFORCE_MODEL_FEATURES", raising=False)
    manager = _fresh_manager()

    with (
        patch("mlflow.pyfunc.load_model", return_value=MagicMock()),
        patch("mlflow.MlflowClient"),
        patch.object(manager, "_get_production_version", return_value="v9"),
        patch.object(
            manager,
            "_load_required_features_artifact",
            return_value=["known_feature", "unknown_feature"],
        ),
        patch(
            "features.registry.FeatureRegistry.list_features",
            return_value=["known_feature"],
        ),
        patch.object(manager, "_load_feature_schema_hash_artifact", return_value=None),
        patch.object(manager, "_load_calibrator_artifact", return_value=(None, False)),
        patch.object(
            manager, "_load_baseline_distribution_artifact", return_value=None
        ),
        patch.object(manager, "_benchmark_inference"),
    ):
        assert manager.load_production_model() is True
        diag = manager.get_diagnostics()
        assert diag["last_reload_status"] == "success"


def test_missing_registry_features_fail_when_strict_flag_set(monkeypatch):
    monkeypatch.setenv("ENFORCE_MODEL_FEATURES", "true")
    manager = _fresh_manager()

    with (
        patch("mlflow.pyfunc.load_model", return_value=MagicMock()),
        patch("mlflow.MlflowClient"),
        patch.object(manager, "_get_production_version", return_value="v9"),
        patch.object(
            manager,
            "_load_required_features_artifact",
            return_value=["known_feature", "unknown_feature"],
        ),
        patch(
            "features.registry.FeatureRegistry.list_features",
            return_value=["known_feature"],
        ),
        patch.object(manager, "_load_feature_schema_hash_artifact", return_value=None),
        patch.object(manager, "_load_calibrator_artifact", return_value=(None, False)),
        patch.object(
            manager, "_load_baseline_distribution_artifact", return_value=None
        ),
        patch.object(manager, "_load_fallback_model", return_value=None),
    ):
        assert manager.load_production_model() is False
        diag = manager.get_diagnostics()
        assert diag["last_reload_status"] == "failed"


def test_ml_health_summary_is_stable_and_bounded():
    manager = _fresh_manager()
    manager.update_feature_coverage_warning(active=True, observed_ts=111.5)

    mock_cache = SimpleNamespace(
        _cache=SimpleNamespace(
            computed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            result={
                "reference_resolution": {
                    "resolution_strategy": "production_stage",
                    "selected_run_id": "run-123",
                },
                "error_code": "no_reference_data",
            },
        )
    )

    with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
        diagnostics = manager.get_diagnostics()
        health_from_method = manager.get_ml_health_summary()

    health = diagnostics["ml_health"]
    assert set(health.keys()) == {
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
    assert health == health_from_method

    assert set(health["model"].keys()) == {
        "state",
        "active_model_version",
        "last_reload_status",
        "last_reload_ts",
        "schema_mismatch_detected",
    }
    assert set(health["benchmark"].keys()) == {
        "enabled",
        "last_status",
        "last_run_ts",
    }
    assert set(health["drift"].keys()) == {
        "reference_resolution_mode",
        "last_error_code",
    }
    assert set(health["feature_coverage"].keys()) == {
        "last_ratio",
        "below_threshold",
    }

    assert isinstance(health["state"], str)
    assert isinstance(health["warnings"], list)
    assert health["status"] in OPERABILITY_STATUSES
    assert isinstance(health["active_model_version"], str)
    assert isinstance(health["last_reload_status"], str)
    assert health["last_reload_ts"] is None or isinstance(
        health["last_reload_ts"], float
    )
    assert isinstance(health["schema_mismatch_detected"], bool)
    assert health["benchmark_status"] is None or isinstance(
        health["benchmark_status"], str
    )
    assert isinstance(health["benchmark"]["enabled"], bool)
    assert health["benchmark"]["last_status"] is None or isinstance(
        health["benchmark"]["last_status"], str
    )
    assert health["benchmark"]["last_run_ts"] is None or isinstance(
        health["benchmark"]["last_run_ts"], float
    )
    assert health["feature_coverage_status"] == "warning"
    assert isinstance(health["feature_coverage_last_seen_ts"], float)
    assert health["feature_coverage"]["last_ratio"] is None or isinstance(
        health["feature_coverage"]["last_ratio"], float
    )
    assert isinstance(health["feature_coverage"]["below_threshold"], bool)
    assert isinstance(health["drift_reference_available"], bool)
    assert health["drift_resolution_mode"] in {"alias", "stage", "latest", "none"}
    assert health["drift"]["reference_resolution_mode"] in {
        "alias",
        "stage",
        "latest",
        "none",
    }
    assert isinstance(health["drift_last_computed_ts"], float)
    assert isinstance(health["drift_last_error_code"], str)
    assert isinstance(health["drift"]["last_error_code"], str)
    assert health["config"] == {
        "strict_feature_schema": False,
        "strict_tuning_resume_validation": False,
        "strict_split_strategy_validation": False,
    }
    assert health["model"]["state"] == health["state"]
    assert health["model"]["active_model_version"] == health["active_model_version"]
    assert health["model"]["last_reload_status"] == health["last_reload_status"]
    assert health["model"]["last_reload_ts"] == health["last_reload_ts"]
    assert (
        health["model"]["schema_mismatch_detected"]
        == health["schema_mismatch_detected"]
    )
    assert all(
        not isinstance(value, list | tuple | set)
        for key, value in health.items()
        if key
        not in {"config", "model", "benchmark", "drift", "feature_coverage", "warnings"}
    )


def test_ml_health_summary_uses_null_for_unset_optional_fields():
    manager = _fresh_manager()
    mock_cache = SimpleNamespace(_cache=None)

    with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
        health = manager.get_ml_health_summary()

    assert health["model"]["last_reload_ts"] is None
    assert health["benchmark"]["last_status"] is None
    assert health["benchmark"]["last_run_ts"] is None
    assert health["feature_coverage"]["last_ratio"] is None
    assert health["feature_coverage_last_seen_ts"] is None
    assert health["drift"]["last_error_code"] is None
    assert health["drift_reference_available"] is None
    assert health["drift_last_computed_ts"] is None
    assert health["drift_last_error_code"] is None


def test_ml_health_summary_rebuilds_canonical_shape_when_payload_incomplete():
    manager = _fresh_manager()
    incomplete = {
        "state": "ready",
        "active_model_version": "v" * 200,
        "last_reload_status": "status-" * 20,
        "last_reload_ts": "nan",
        "schema_mismatch_detected": False,
        "benchmark_last_status": "benchmark-status-" * 20,
        "benchmark_last_run_ts": "inf",
        "feature_coverage_warning_active": True,
        "feature_coverage_last_ratio": 1.7,
        "feature_coverage_warning_last_seen_ts": "not-a-float",
        "config": {
            "strict_feature_schema": "yes",
            "strict_tuning_resume_validation": "0",
            "strict_split_strategy_validation": 1,
            "unexpected_key": True,
        },
        "ml_health": {"model": [], "unexpected_nested": {"a": "b"}},
    }

    with (
        patch.object(manager, "get_diagnostics", return_value=incomplete),
        patch(
            "forecast.drift_cache.get_drift_cache",
            return_value=SimpleNamespace(_cache=None),
        ),
    ):
        health = manager.get_ml_health_summary()

    assert set(health["config"].keys()) == {
        "strict_feature_schema",
        "strict_tuning_resume_validation",
        "strict_split_strategy_validation",
    }
    assert health["config"] == {
        "strict_feature_schema": True,
        "strict_tuning_resume_validation": False,
        "strict_split_strategy_validation": True,
    }
    assert set(health["model"].keys()) == {
        "state",
        "active_model_version",
        "last_reload_status",
        "last_reload_ts",
        "schema_mismatch_detected",
    }
    assert health["status"] in OPERABILITY_STATUSES
    assert set(health["benchmark"].keys()) == {"enabled", "last_status", "last_run_ts"}
    assert set(health["drift"].keys()) == {
        "reference_resolution_mode",
        "last_error_code",
    }
    assert set(health["feature_coverage"].keys()) == {"last_ratio", "below_threshold"}
    assert len(health["active_model_version"]) <= 64
    assert len(health["last_reload_status"]) <= 32
    assert health["last_reload_ts"] is None
    assert health["benchmark"]["last_run_ts"] is None
    assert health["feature_coverage_last_seen_ts"] is None
    assert health["benchmark"]["last_status"] is None or (
        len(health["benchmark"]["last_status"]) <= 32
    )
    assert health["feature_coverage"]["last_ratio"] == 1.0
    assert all(
        not isinstance(value, list | tuple | set)
        for key, value in health.items()
        if key
        not in {"model", "benchmark", "drift", "feature_coverage", "config", "warnings"}
    )


def test_ml_health_feature_coverage_ratio_tracks_last_observation():
    manager = _fresh_manager()
    manager.update_feature_coverage_warning(
        active=True,
        coverage_ratio=1.25,
        observed_ts=123.0,
    )

    diagnostics = manager.get_diagnostics()
    health = diagnostics["ml_health"]

    assert diagnostics["feature_coverage_last_ratio"] == 1.0
    assert health["feature_coverage"]["last_ratio"] == 1.0
    assert health["feature_coverage"]["below_threshold"] is True

    manager.update_feature_coverage_warning(active=False, coverage_ratio=-0.2)
    refreshed = manager.get_ml_health_summary()

    assert refreshed["feature_coverage"]["last_ratio"] == 0.0
    assert refreshed["feature_coverage"]["below_threshold"] is False
