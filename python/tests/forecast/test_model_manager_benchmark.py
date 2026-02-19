from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from forecast.model_manager import ModelManager, ModelStateBundle


def _assert_rng_state_equal(before, after):
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2] == after[2]
    assert before[3] == after[3]
    assert before[4] == after[4]


def _make_bundle(model_version: str) -> ModelStateBundle:
    return ModelStateBundle(
        model=MagicMock(),
        version=model_version,
        source="mlflow",
        required_features=["f1", "f2"],
        calibrator=None,
        calibrator_loaded=False,
        baseline_distribution=None,
        feature_importance=None,
    )


def test_benchmark_matches_version_after_stripping_v_prefix():
    manager = ModelManager()
    bundle = _make_bundle("v7")
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        SimpleNamespace(current_stage="Production", version="7", run_id="run-7"),
    ]

    with patch("mlflow.MlflowClient", return_value=mock_client):
        manager._benchmark_inference(bundle, n_samples=3)

    logged_metric_names = {c.args[1] for c in mock_client.log_metric.call_args_list}
    assert logged_metric_names == {
        "inference_latency_p50_ms",
        "inference_latency_p95_ms",
        "inference_latency_p99_ms",
    }


def test_benchmark_does_not_advance_global_numpy_rng_state():
    manager = ModelManager()
    bundle = _make_bundle("v1")
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        SimpleNamespace(current_stage="Production", version="1", run_id="run-1"),
    ]

    np.random.seed(12345)
    before_state = np.random.get_state()

    with patch("mlflow.MlflowClient", return_value=mock_client):
        manager._benchmark_inference(bundle, n_samples=5)

    after_state = np.random.get_state()
    _assert_rng_state_equal(before_state, after_state)
