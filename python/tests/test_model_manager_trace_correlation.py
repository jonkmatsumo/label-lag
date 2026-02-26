"""Tests for model reload trace correlation metadata."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from forecast.model_manager import ModelManager


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


def test_load_training_identity_artifact_reads_json(tmp_path):
    """Model manager should load training identity artifacts for production version."""
    manager = _fresh_manager()
    identity_path = tmp_path / "training_run_identity.json"
    expected_identity = {
        "schema_version": 1,
        "mlflow_run_id": "run-123",
        "model_version": "7",
        "feature_schema_hash": "abc123",
    }
    identity_path.write_text(json.dumps(expected_identity))

    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        SimpleNamespace(current_stage="Production", version="7", run_id="run-123"),
    ]
    mock_client.download_artifacts.return_value = str(identity_path)

    with patch("mlflow.MlflowClient", return_value=mock_client):
        loaded = manager._load_training_run_identity_artifact("v7")

    assert loaded == expected_identity
    mock_client.download_artifacts.assert_called_once_with(
        "run-123", "training_run_identity.json"
    )


def test_reload_span_and_diagnostics_include_training_identity():
    """Reload span attributes and diagnostics should expose training identity."""
    manager = _fresh_manager()
    mock_span = MagicMock()
    span_context = MagicMock()
    span_context.__enter__.return_value = mock_span
    span_context.__exit__.return_value = False

    identity = {
        "schema_version": 1,
        "mlflow_run_id": "train-run-42",
        "model_version": "11",
        "feature_schema_hash": "hash-xyz",
    }

    with (
        patch.dict(
            "os.environ",
            {"INFERENCE_MODEL_RELOAD_SPAN_ENABLED": "true"},
            clear=False,
        ),
        patch("mlflow.start_span", return_value=span_context),
        patch("mlflow.pyfunc.load_model", return_value=MagicMock()),
        patch.object(manager, "_get_production_version", return_value="v11"),
        patch.object(manager, "_load_required_features_artifact", return_value=["f1"]),
        patch.object(manager, "_load_feature_schema_hash_artifact", return_value=None),
        patch.object(manager, "_load_calibrator_artifact", return_value=(None, False)),
        patch.object(
            manager, "_load_baseline_distribution_artifact", return_value=None
        ),
        patch.object(
            manager,
            "_load_training_run_identity_artifact",
            return_value=identity,
        ),
        patch.object(manager, "_benchmark_inference"),
    ):
        assert manager.load_production_model() is True

    mock_span.set_attribute.assert_any_call("model.reload.status", "loaded_from_mlflow")
    mock_span.set_attribute.assert_any_call("ml.model.version", "v11")
    mock_span.set_attribute.assert_any_call("ml.training.run_id", "train-run-42")
    mock_span.set_attribute.assert_any_call("ml.model.version", "11")
    mock_span.set_attribute.assert_any_call("ml.feature.schema_hash", "hash-xyz")

    diagnostics = manager.get_diagnostics()
    assert diagnostics["ml.training.run_id"] == "train-run-42"
    assert diagnostics["ml.model.version"] == "11"
    assert diagnostics["ml.feature.schema_hash"] == "hash-xyz"


def test_missing_training_identity_is_non_fatal_and_silent(caplog):
    """Missing training identity artifact should not cause errors or warnings."""
    manager = _fresh_manager()
    mock_span = MagicMock()
    span_context = MagicMock()
    span_context.__enter__.return_value = mock_span
    span_context.__exit__.return_value = False

    mock_model = MagicMock()
    mock_model.feature_importances_ = [0.1]  # Match required_features length

    with (
        patch.dict(
            "os.environ",
            {"INFERENCE_MODEL_RELOAD_SPAN_ENABLED": "true"},
            clear=False,
        ),
        patch("mlflow.start_span", return_value=span_context),
        patch("mlflow.pyfunc.load_model", return_value=mock_model),
        patch.object(manager, "_get_production_version", return_value="v12"),
        patch.object(manager, "_load_required_features_artifact", return_value=["f1"]),
        patch(
            "features.registry.FeatureRegistry.list_features", return_value=["f1"]
        ),  # Silence feature warning
        patch.object(manager, "_load_feature_schema_hash_artifact", return_value=None),
        patch.object(manager, "_load_calibrator_artifact", return_value=(None, False)),
        patch.object(
            manager, "_load_baseline_distribution_artifact", return_value=None
        ),
        patch.object(
            manager,
            "_load_training_run_identity_artifact",
            return_value=None,  # Handled absence
        ),
        patch.object(manager, "_benchmark_inference"),
    ):
        with caplog.at_level("WARNING"):
            assert manager.load_production_model() is True

    # Assert no warnings or errors in logs
    for record in caplog.records:
        assert record.levelno < 30  # Less than WARNING

    # Assert span was created but without identity attributes
    mock_span.set_attribute.assert_any_call("ml.model.version", "v12")
    # Verify ml.training.* attributes were NOT set
    for call in mock_span.set_attribute.call_args_list:
        attr_key = str(call[0][0])
        assert not attr_key.startswith("ml.training")
        if attr_key.startswith("ml."):
            assert attr_key == "ml.model.version"
