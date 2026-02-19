"""Tests for model-manager inference benchmark behavior."""

from unittest.mock import MagicMock, patch

import numpy as np

from forecast.model_manager import ModelManager


class _PredictModel:
    def predict(self, features):
        return np.zeros(len(features))


class _FailingPredictModel:
    def predict(self, features):
        raise RuntimeError("predict failed")


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    manager = ModelManager()
    return manager


def test_load_benchmark_does_not_log_metrics_to_mlflow_run():
    """Model load benchmark must not mutate historical MLflow runs."""
    manager = _fresh_manager()

    with (
        patch("mlflow.pyfunc.load_model", return_value=_PredictModel()),
        patch("mlflow.MlflowClient") as mock_client_cls,
        patch.object(manager, "_get_production_version", return_value="1"),
        patch.object(manager, "_load_required_features_artifact"),
        patch.object(manager, "_load_calibrator_artifact"),
        patch.object(manager, "_load_baseline_distribution_artifact"),
        patch.object(
            manager,
            "_benchmark_inference",
            wraps=manager._benchmark_inference,
        ) as benchmark_spy,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        assert manager.load_production_model() is True
        assert benchmark_spy.called
        mock_client.log_metric.assert_not_called()


def test_benchmark_failures_are_non_blocking():
    """Benchmark exceptions should be swallowed and never break model load."""
    manager = _fresh_manager()
    manager._model = _FailingPredictModel()
    manager._required_features = ["a", "b", "c"]

    # Should not raise even if predict fails inside benchmark loop.
    manager._benchmark_inference(n_samples=2)
