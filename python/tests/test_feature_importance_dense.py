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


def test_feature_importance_mismatch_returns_best_effort():
    """Best-effort mapping when model importances count != required features."""
    manager = ModelManager()
    manager._initialized = False
    manager.__init__()

    # Model has 2 importances, but we require 3 features
    class _MismatchModel:
        def __init__(self):
            self.feature_importances_ = np.array([0.9, 0.1])

    manager._model = _MismatchModel()
    manager._required_features = ["feat_a", "feat_b", "feat_c"]

    importance = manager.get_feature_importance()

    assert importance is not None
    assert len(importance) == 3
    assert importance["feat_a"] == 0.9
    assert importance["feat_b"] == 0.1
    assert importance["feat_c"] == 0.0  # Padded with zero


def test_feature_importance_extra_model_features():
    """Ensure best-effort mapping when model has MORE features than required."""
    manager = ModelManager()
    manager._initialized = False
    manager.__init__()

    # Model has 4 importances, but we only require 2 features
    class _ExtraModel:
        def __init__(self):
            self.feature_importances_ = np.array([0.7, 0.2, 0.1, 0.0])

    manager._model = _ExtraModel()
    manager._required_features = ["feat_a", "feat_b"]

    importance = manager.get_feature_importance()

    assert importance is not None
    assert len(importance) == 2
    assert importance["feat_a"] == 0.7
    assert importance["feat_b"] == 0.2
