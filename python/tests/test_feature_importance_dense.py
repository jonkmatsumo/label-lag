"""Regression tests for dense feature importance extraction."""

import numpy as np

from forecast.model_manager import ModelManager


class _DenseImportanceModel:
    def __init__(self):
        self.feature_importances_ = np.array([0.8, 0.2, 0.0])


class _PyfuncLikeWrapper:
    def __init__(self):
        self._model_impl = _DenseImportanceModel()


def test_dense_feature_importance_includes_zero_usage_features():
    """Feature importance returns all required features, including unused ones."""
    manager = ModelManager()
    manager._initialized = False
    manager.__init__()

    manager._model = _PyfuncLikeWrapper()
    manager._required_features = ["feat_a", "feat_b", "feat_c"]

    importance = manager.get_feature_importance()

    assert importance is not None
    assert set(importance.keys()) == {"feat_a", "feat_b", "feat_c"}
    assert importance["feat_a"] == 0.8
    assert importance["feat_b"] == 0.2
    assert importance["feat_c"] == 0.0
