"""Tests for hyperparameter passthrough and early stopping."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from model.loader import DataLoader, TrainTestSplit
from model.train import train_model


class TestHyperparamsPassedToXGBoost:
    """Hyperparameters are passed to XGBClassifier."""

    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    @patch("model.train._get_git_sha", return_value="x")
    def test_hyperparams_passed_to_xgboost(
        self, _mock_git, mock_loader_cls, mock_mlflow
    ):
        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = DataLoader.FEATURE_COLUMNS
        mock_loader_cls.return_value = mock_loader
        mock_split = TrainTestSplit(
            X_train=pd.DataFrame(
                {
                    "a": [1, 2, 3, 4],
                    "b": [1.0, 2.0, 3.0, 4.0],
                    "c": [0.1, 0.2, 0.3, 0.4],
                }
            ),
            y_train=pd.Series([0, 1, 0, 1]),
            X_test=pd.DataFrame({"a": [5], "b": [5.0], "c": [0.5]}),
            y_test=pd.Series([1]),
        )
        mock_loader.load_train_test_split.return_value = mock_split
        mock_run = MagicMock()
        mock_run.info.run_id = "r"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.set_experiment.return_value = None

        with patch("model.train.XGBClassifier") as mock_xgb:
            mock_xgb.return_value.fit.return_value = None
            mock_xgb.return_value.predict.return_value = np.array([1])
            mock_xgb.return_value.predict_proba.return_value = np.array([[0.2, 0.8]])
            mock_xgb.return_value.get_booster.return_value.attr.return_value = None
            train_model(
                n_estimators=200,
                learning_rate=0.05,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.9,
                gamma=0.5,
                reg_alpha=0.1,
                reg_lambda=2.0,
                random_state=99,
                feature_columns=["a", "b", "c"],
            )
            call_kw = mock_xgb.call_args[1]
            assert call_kw["n_estimators"] == 200
            assert call_kw["learning_rate"] == 0.05
            assert call_kw["min_child_weight"] == 3
            assert call_kw["subsample"] == 0.8
            assert call_kw["colsample_bytree"] == 0.9
            assert call_kw["gamma"] == 0.5
            assert call_kw["reg_alpha"] == 0.1
            assert call_kw["reg_lambda"] == 2.0
            assert call_kw["random_state"] == 99


class TestHyperparamsLoggedToMlflow:
    """Hyperparameters are logged to MLflow."""

    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    @patch("model.train._get_git_sha", return_value="x")
    def test_hyperparams_logged_to_mlflow(
        self, _mock_git, mock_loader_cls, mock_mlflow
    ):
        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = DataLoader.FEATURE_COLUMNS
        mock_loader_cls.return_value = mock_loader
        mock_split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": [0.1, 0.2]}),
            y_train=pd.Series([0, 1]),
            X_test=pd.DataFrame({"a": [3], "b": [3.0], "c": [0.3]}),
            y_test=pd.Series([0]),
        )
        mock_loader.load_train_test_split.return_value = mock_split
        mock_run = MagicMock()
        mock_run.info.run_id = "r"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.set_experiment.return_value = None

        train_model(
            n_estimators=150,
            learning_rate=0.02,
            random_state=123,
            feature_columns=["a", "b", "c"],
        )
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["n_estimators"] == 150
        assert params["learning_rate"] == 0.02
        assert params["random_state"] == 123


class TestEarlyStopping:
    """Early stopping behavior."""

    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    @patch("model.train._get_git_sha", return_value="x")
    def test_early_stopping_logs_best_iteration(
        self, _mock_git, mock_loader_cls, mock_mlflow
    ):
        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = DataLoader.FEATURE_COLUMNS
        mock_loader_cls.return_value = mock_loader
        n = 60
        mock_split = TrainTestSplit(
            X_train=pd.DataFrame({"a": list(range(n)), "b": [1.0] * n, "c": [0.1] * n}),
            y_train=pd.Series([0] * 45 + [1] * 15),
            X_test=pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": [0.1, 0.2]}),
            y_test=pd.Series([0, 1]),
        )
        mock_loader.load_train_test_split.return_value = mock_split
        mock_run = MagicMock()
        mock_run.info.run_id = "r"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.set_experiment.return_value = None

        train_model(
            early_stopping_rounds=5,
            feature_columns=["a", "b", "c"],
        )
        metric_calls = {c[0][0]: c[0][1] for c in mock_mlflow.log_metric.call_args_list}
        assert "best_iteration" in metric_calls


class TestDefaultHyperparamsUnchanged:
    """Default hyperparameters match schema."""

    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    @patch("model.train._get_git_sha", return_value="x")
    def test_default_hyperparams_unchanged(
        self, _mock_git, mock_loader_cls, mock_mlflow
    ):
        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = DataLoader.FEATURE_COLUMNS
        mock_loader_cls.return_value = mock_loader
        mock_split = TrainTestSplit(
            X_train=pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": [0.1, 0.2]}),
            y_train=pd.Series([0, 1]),
            X_test=pd.DataFrame({"a": [3], "b": [3.0], "c": [0.3]}),
            y_test=pd.Series([0]),
        )
        mock_loader.load_train_test_split.return_value = mock_split
        mock_run = MagicMock()
        mock_run.info.run_id = "r"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.set_experiment.return_value = None

        train_model(feature_columns=["a", "b", "c"])
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["n_estimators"] == 100
        assert params["learning_rate"] == 0.1
        assert params["random_state"] == 42
        assert "early_stopping_rounds" not in params
