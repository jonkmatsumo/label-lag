"""Guardrail tests for ModelManager default/warn-only behavior."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from forecast.model_manager import ModelManager
from training.reason_codes import MODEL_MANAGER_BASELINE_DIAGNOSTIC_KEYS


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

    health = diagnostics["ml_health"]
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
    }
    assert isinstance(health["state"], str)
    assert isinstance(health["active_model_version"], str)
    assert isinstance(health["last_reload_status"], str)
    assert health["last_reload_ts"] is None or isinstance(
        health["last_reload_ts"], float
    )
    assert isinstance(health["schema_mismatch_detected"], bool)
    assert health["benchmark_status"] is None or isinstance(
        health["benchmark_status"], str
    )
    assert health["feature_coverage_status"] == "warning"
    assert isinstance(health["feature_coverage_last_seen_ts"], float)
    assert isinstance(health["drift_reference_available"], bool)
    assert health["drift_resolution_mode"] in {"alias", "stage", "latest", "none"}
    assert isinstance(health["drift_last_computed_ts"], float)
    assert isinstance(health["drift_last_error_code"], str)
    assert all(
        not isinstance(value, list | tuple | set | dict) for value in health.values()
    )
