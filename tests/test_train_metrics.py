"""Unit tests for expanded training metrics and artifact generation."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from model.loader import DataLoader, TrainTestSplit
from model.train import (
    _generate_model_card,
    _save_confusion_matrix_plot,
    _save_feature_importance,
    train_model,
)


class TestExpandedMetricsComputed:
    """Verify all 9+ metrics are computed and logged."""

    @patch("model.train._get_git_sha", return_value="abc123")
    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    def test_expanded_metrics_computed(self, mock_loader_cls, mock_mlflow, _mock_git):
        """Verify f1, roc_auc, log_loss, brier, tp, fp, tn, fn are logged."""
        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = DataLoader.FEATURE_COLUMNS
        mock_loader_cls.return_value = mock_loader

        mock_split = TrainTestSplit(
            X_train=pd.DataFrame(
                {
                    "velocity_24h": [1, 2, 3, 4],
                    "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0, 2.5],
                    "balance_volatility_z_score": [0.0, 0.5, 1.0, 1.5],
                }
            ),
            y_train=pd.Series([0, 1, 0, 1]),
            X_test=pd.DataFrame(
                {
                    "velocity_24h": [5, 6],
                    "amount_to_avg_ratio_30d": [3.0, 3.5],
                    "balance_volatility_z_score": [2.0, 2.5],
                }
            ),
            y_test=pd.Series([1, 0]),
        )
        mock_loader.load_train_test_split.return_value = mock_split

        mock_run = MagicMock()
        mock_run.info.run_id = "run_xyz"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.set_experiment.return_value = None

        train_model()

        assert mock_mlflow.log_metrics.called
        metrics = mock_mlflow.log_metrics.call_args[0][0]
        required = {
            "precision",
            "recall",
            "pr_auc",
            "f1",
            "roc_auc",
            "log_loss",
            "brier_score",
            "tp",
            "fp",
            "tn",
            "fn",
        }
        for k in required:
            assert k in metrics, f"missing metric {k}"
        assert len(metrics) >= 11


class TestConfusionMatrixCountsCorrect:
    """Verify TP/FP/TN/FN from confusion matrix."""

    @patch("model.train._get_git_sha", return_value="abc")
    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    def test_confusion_matrix_counts_correct(self, mock_loader_cls, mock_mlflow, _git):
        """tp, fp, tn, fn non-negative and sum to test size."""
        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = ["v", "a", "b"]
        mock_loader_cls.return_value = mock_loader

        np.random.seed(42)
        n_train, n_test = 20, 10
        x_train = pd.DataFrame(np.random.randn(n_train, 3), columns=["v", "a", "b"])
        y_train = pd.Series(np.random.randint(0, 2, n_train))
        x_test = pd.DataFrame(np.random.randn(n_test, 3), columns=["v", "a", "b"])
        y_test = pd.Series(np.random.randint(0, 2, n_test))
        mock_split = TrainTestSplit(
            X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test
        )
        mock_loader.load_train_test_split.return_value = mock_split

        mock_run = MagicMock()
        mock_run.info.run_id = "r"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.set_experiment.return_value = None

        train_model(feature_columns=["v", "a", "b"])

        metrics = mock_mlflow.log_metrics.call_args[0][0]
        tp, fp, tn, fn = metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]
        assert tp >= 0 and fp >= 0 and tn >= 0 and fn >= 0
        assert tp + fp + tn + fn == len(y_test)


class TestFeatureImportanceJsonStructure:
    """Verify feature importance JSON artifact format."""

    def test_feature_importance_json_structure(self):
        """_save_feature_importance produces valid JSON with feature -> float."""
        mock_clf = MagicMock()
        mock_clf.feature_importances_ = np.array([0.5, 0.3, 0.2])
        names = ["f1", "f2", "f3"]
        with tempfile.TemporaryDirectory() as d:
            base = Path(d) / "fi"
            jpath, _ = _save_feature_importance(mock_clf, names, base)
            with open(jpath) as f:
                data = json.load(f)
            assert list(data.keys()) == names
            for v in data.values():
                assert isinstance(v, (int, float))


class TestModelCardContainsRequiredSections:
    """Verify model card markdown structure."""

    def test_model_card_contains_required_sections(self):
        """Model card has Training Summary, Metrics, Config."""
        params = {
            "train_size": 100,
            "test_size": 20,
            "train_fraud_rate": 0.05,
            "test_fraud_rate": 0.04,
            "max_depth": 6,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "training_window_days": 30,
        }
        metrics = {"precision": 0.8, "recall": 0.7, "pr_auc": 0.75, "f1": 0.74}
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            _generate_model_card(params, metrics, path)
            text = Path(path).read_text()
            assert "## Training Summary" in text
            assert "## Metrics" in text
            assert "## Config" in text
            assert "Train size" in text
            assert "precision" in text
            assert "max_depth" in text
        finally:
            Path(path).unlink(missing_ok=True)


class TestConfusionMatrixPlot:
    """Verify confusion matrix plot helper."""

    def test_save_confusion_matrix_plot_creates_file(self):
        """_save_confusion_matrix_plot writes a PNG file."""
        y_test = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cm.png"
            _save_confusion_matrix_plot(y_test, y_pred, path)
            assert path.exists()
            assert path.stat().st_size > 0
