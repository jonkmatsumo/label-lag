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


def test_benchmark_metrics_failure_does_not_block_load():
    """Ensure that if Prometheus metrics fail, the model load still succeeds."""
    manager = _fresh_manager()

    # Setup a mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5])
    manager._model = mock_model
    manager._model_source = "mlflow"
    manager._model_version = "v1"
    manager._required_features = ["feat_a"]

    # Mock metrics to raise an exception
    with patch("forecast.metrics.inference_benchmark_sample_latency_ms") as mock_hist:
        mock_hist.observe.side_effect = Exception("Prometheus down")

        # This should NOT raise an exception
        manager._benchmark_inference(n_samples=5)

    # Verify it was marked as benchmarked anyway
    assert "v1" in manager._benchmarked_versions


def test_benchmark_gating_prevents_repeated_runs():
    """Ensure benchmarking only happens once per version."""
    manager = _fresh_manager()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5])
    manager._model = mock_model
    manager._model_version = "v2"
    manager._required_features = ["feat_a"]

    # First call
    manager._benchmark_inference(n_samples=1)
    assert "v2" in manager._benchmarked_versions
    assert mock_model.predict.call_count == 1

    # Second call - should skip
    manager._benchmark_inference(n_samples=1)
    # Call count stays 1
    assert mock_model.predict.call_count == 1


def test_benchmark_skipped_when_disabled():
    """Benchmark should skip all model calls when explicitly disabled."""
    manager = _fresh_manager()

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5])
    manager._model = mock_model
    manager._model_source = "mlflow"
    manager._model_version = "v3"
    manager._required_features = ["feat_a"]

    with patch("forecast.model_manager.INFERENCE_BENCHMARK_ENABLED", False):
        manager._benchmark_inference(n_samples=3)

    assert mock_model.predict.call_count == 0
    assert manager._benchmark_last_status == "skipped_disabled"
    assert "v3" not in manager._benchmarked_versions


def test_benchmark_sample_rate_gating_with_seeded_rng():
    """Sample-rate gating should be deterministic with injected seeded RNG."""
    seed = 1234
    sample_draw = float(np.random.default_rng(seed).random())

    # Skip case: threshold lower than deterministic draw.
    skip_manager = _fresh_manager()
    skip_model = MagicMock()
    skip_model.predict.return_value = np.array([0.5])
    skip_manager._model = skip_model
    skip_manager._model_source = "mlflow"
    skip_manager._model_version = "v4"
    skip_manager._required_features = ["feat_a"]

    with (
        patch("forecast.model_manager.INFERENCE_BENCHMARK_ENABLED", True),
        patch(
            "forecast.model_manager.INFERENCE_BENCHMARK_SAMPLE_RATE",
            max(0.0, sample_draw - 1e-6),
        ),
    ):
        skip_manager._benchmark_inference(
            n_samples=1,
            sample_rng=np.random.default_rng(seed),
        )

    assert skip_model.predict.call_count == 0
    assert skip_manager._benchmark_last_status == "skipped_sampled_out"

    # Run case: threshold higher than deterministic draw.
    run_manager = _fresh_manager()
    run_model = MagicMock()
    run_model.predict.return_value = np.array([0.5])
    run_manager._model = run_model
    run_manager._model_source = "mlflow"
    run_manager._model_version = "v5"
    run_manager._required_features = ["feat_a"]

    with (
        patch("forecast.model_manager.INFERENCE_BENCHMARK_ENABLED", True),
        patch(
            "forecast.model_manager.INFERENCE_BENCHMARK_SAMPLE_RATE",
            min(1.0, sample_draw + 1e-6),
        ),
    ):
        run_manager._benchmark_inference(
            n_samples=1,
            sample_rng=np.random.default_rng(seed),
        )

    assert run_model.predict.call_count == 1
    assert run_manager._benchmark_last_status == "success"
