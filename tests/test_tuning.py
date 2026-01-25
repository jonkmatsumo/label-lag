"""Unit tests for Optuna tuning module."""

from unittest.mock import MagicMock, patch

import pandas as pd

from model.tuning import (
    DEFAULT_SEARCH_SPACE,
    run_tuning_study,
)


class TestTuningStudy:
    """Tests for run_tuning_study."""

    def test_tuning_runs_n_trials(self):
        """Study runs requested number of trials."""
        n = 60
        x = pd.DataFrame({"a": list(range(n)), "b": [1.0] * n, "c": [0.1] * n})
        y = pd.Series([0] * 45 + [1] * 15)
        x_val = x.iloc[50:]
        y_val = y.iloc[50:]
        x_tr = x.iloc[:50]
        y_tr = y.iloc[:50]
        best, df = run_tuning_study(
            x_tr, y_tr, x_val, y_val, n_trials=5, metric="pr_auc"
        )
        assert len(df) == 5
        assert "trial" in df.columns
        assert "value" in df.columns

    def test_tuning_returns_best_params(self):
        """Best params dict is returned."""
        n = 60
        x = pd.DataFrame({"a": list(range(n)), "b": [1.0] * n, "c": [0.1] * n})
        y = pd.Series([0] * 45 + [1] * 15)
        x_val = x.iloc[50:]
        y_val = y.iloc[50:]
        x_tr = x.iloc[:50]
        y_tr = y.iloc[:50]
        best, _ = run_tuning_study(
            x_tr, y_tr, x_val, y_val, n_trials=3, metric="pr_auc"
        )
        assert isinstance(best, dict)
        assert "max_depth" in best
        assert "learning_rate" in best

    def test_tuning_respects_timeout(self):
        """Study stops within timeout."""
        n = 80
        x = pd.DataFrame({"a": list(range(n)), "b": [1.0] * n, "c": [0.1] * n})
        y = pd.Series([0, 1] * (n // 2))  # alternate so both in train/val
        x_tr, x_val = x.iloc[:60], x.iloc[60:]
        y_tr, y_val = y.iloc[:60], y.iloc[60:]
        best, df = run_tuning_study(
            x_tr,
            y_tr,
            x_val,
            y_val,
            n_trials=100,
            timeout_seconds=2,
            metric="pr_auc",
        )
        assert len(df) < 100

    def test_disabled_tuning_skipped(self):
        """When tuning disabled, train_model does not run study."""
        from api.schemas import TuningConfig
        from model.loader import DataLoader, TrainTestSplit
        from model.train import train_model

        with patch("model.train.DataLoader") as mock_loader:
            mock_loader.FEATURE_COLUMNS = DataLoader.FEATURE_COLUMNS
            m = MagicMock()
            m.load_train_test_split.return_value = TrainTestSplit(
                X_train=pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": [0.1, 0.2]}),
                y_train=pd.Series([0, 1]),
                X_test=pd.DataFrame({"a": [3], "b": [3.0], "c": [0.3]}),
                y_test=pd.Series([0]),
            )
            mock_loader.return_value = m
            with (
                patch("model.train.mlflow"),
                patch("model.train._get_git_sha", return_value="x"),
            ):
                with patch("model.train.run_tuning_study") as mock_tune:
                    cfg = TuningConfig(enabled=False)
                    train_model(
                        tuning_config=cfg,
                        feature_columns=["a", "b", "c"],
                    )
                    mock_tune.assert_not_called()


class TestSearchSpace:
    """DEFAULT_SEARCH_SPACE structure."""

    def test_search_space_has_expected_keys(self):
        """All XGBoost params in search space."""
        expected = {
            "max_depth",
            "n_estimators",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        }
        assert set(DEFAULT_SEARCH_SPACE.keys()) == expected
