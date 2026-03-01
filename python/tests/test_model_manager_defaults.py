"""Guardrail tests for ModelManager default/warn-only behavior."""

from unittest.mock import MagicMock, patch

from forecast.model_manager import ModelManager
from training.reason_codes import MODEL_MANAGER_BASELINE_DIAGNOSTIC_KEYS


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


def test_diagnostics_snapshot_includes_required_baseline_fields():
    manager = _fresh_manager()
    diag = manager.get_diagnostics()

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
