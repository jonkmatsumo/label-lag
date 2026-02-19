import numpy as np

from model.train import _compute_metrics


class TestCVMetricsRobustness:
    def test_compute_metrics_handles_all_zeros_fold(self):
        """Test metrics computation when validation fold has only negative samples."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.1, 0.3, 0.2])

        metrics = _compute_metrics(y_true, y_pred, y_proba)

        # Undefined metrics should be None
        assert metrics["roc_auc"] is None
        assert metrics["log_loss"] is None

        # Defined metrics
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["pr_auc"] == 0.0
        assert metrics["tp"] == 0
        assert metrics["fp"] == 0

    def test_compute_metrics_handles_all_ones_fold(self):
        """Test metrics computation when validation fold has only positive samples."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 1])
        y_proba = np.array([0.9, 0.8, 0.95])

        metrics = _compute_metrics(y_true, y_pred, y_proba)

        # Undefined metrics should be None
        assert metrics["roc_auc"] is None
        assert metrics["log_loss"] is None

        # Defined metrics
        assert metrics["recall"] == 1.0
        assert metrics["tp"] == 3
        assert metrics["fn"] == 0

    def test_compute_metrics_handles_normal_fold(self):
        """Test metrics computation for a normal mixed fold."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.8, 0.7])

        metrics = _compute_metrics(y_true, y_pred, y_proba)

        assert metrics["roc_auc"] is not None
        assert metrics["log_loss"] is not None
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 2 / 3
        assert metrics["tp"] == 2
        assert metrics["fp"] == 1
