"""Tests for training with feature column selection."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd

from model.loader import DataLoader, TrainTestSplit
from model.train import train_model


class TestTrainModelWithFeatureColumns:
    """Tests for train_model with feature column override."""

    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    def test_train_model_logs_feature_columns_param(self, mock_loader_cls, mock_mlflow):
        """Verify that feature_columns param is logged to MLflow."""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader

        # Mock train/test split
        mock_split = TrainTestSplit(
            X_train=pd.DataFrame(
                {
                    "velocity_24h": [1, 2, 3],
                    "amount_to_avg_ratio_30d": [1.0, 1.5, 2.0],
                }
            ),
            y_train=pd.Series([0, 1, 0]),
            X_test=pd.DataFrame(
                {
                    "velocity_24h": [4, 5],
                    "amount_to_avg_ratio_30d": [2.5, 3.0],
                }
            ),
            y_test=pd.Series([1, 0]),
        )
        mock_loader.load_train_test_split.return_value = mock_split

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.set_experiment.return_value = None

        # Call train_model with custom feature columns
        custom_features = ["velocity_24h", "amount_to_avg_ratio_30d"]
        train_model(feature_columns=custom_features)

        # Verify feature_columns was passed to loader
        call_args = mock_loader.load_train_test_split.call_args
        assert call_args.kwargs["feature_columns"] == custom_features

        # Verify mlflow.log_params was called with feature_columns
        assert mock_mlflow.log_params.called
        params_call = mock_mlflow.log_params.call_args[0][0]
        assert "feature_columns" in params_call
        assert json.loads(params_call["feature_columns"]) == custom_features

    @patch("model.train._get_git_sha", return_value="test_sha")
    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    @patch("model.train.os.path.join")
    @patch("model.train.open", create=True)
    def test_train_model_logs_feature_columns_artifact(
        self, mock_open, mock_join, mock_loader_cls, mock_mlflow, _mock_git_sha
    ):
        """Verify that feature_columns.json artifact is logged."""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader_cls.return_value = mock_loader

        # Mock train/test split
        mock_split = TrainTestSplit(
            X_train=pd.DataFrame({"velocity_24h": [1, 2, 3]}),
            y_train=pd.Series([0, 1, 0]),
            X_test=pd.DataFrame({"velocity_24h": [4, 5]}),
            y_test=pd.Series([1, 0]),
        )
        mock_loader.load_train_test_split.return_value = mock_split

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.set_experiment.return_value = None

        # Mock tempfile and file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_join.side_effect = lambda *args: "/tmp/" + args[-1]

        # Call train_model with custom feature columns
        custom_features = ["velocity_24h"]
        train_model(feature_columns=custom_features)

        # Verify mlflow.log_artifact was called
        assert mock_mlflow.log_artifact.called

        # Check that feature_columns.json was written
        artifact_calls = [
            call[0][0] for call in mock_mlflow.log_artifact.call_args_list
        ]
        feature_columns_artifact = any(
            "feature_columns.json" in str(call) for call in artifact_calls
        )
        assert feature_columns_artifact, (
            "feature_columns.json artifact should be logged"
        )

    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    def test_train_model_uses_default_columns_when_none_provided(
        self, mock_loader_cls, mock_mlflow
    ):
        """Verify that default FEATURE_COLUMNS are used when feature_columns is None."""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = DataLoader.FEATURE_COLUMNS
        mock_loader_cls.return_value = mock_loader

        # Mock train/test split
        mock_split = TrainTestSplit(
            X_train=pd.DataFrame(
                {
                    "velocity_24h": [1, 2],
                    "amount_to_avg_ratio_30d": [1.0, 1.5],
                    "balance_volatility_z_score": [0.0, 0.5],
                }
            ),
            y_train=pd.Series([0, 1]),
            X_test=pd.DataFrame(
                {
                    "velocity_24h": [3],
                    "amount_to_avg_ratio_30d": [2.0],
                    "balance_volatility_z_score": [1.0],
                }
            ),
            y_test=pd.Series([0]),
        )
        mock_loader.load_train_test_split.return_value = mock_split

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.set_experiment.return_value = None

        # Call train_model without feature_columns
        train_model()

        # Verify loader was called with feature_columns=None
        call_args = mock_loader.load_train_test_split.call_args
        assert call_args.kwargs.get("feature_columns") is None

        # Verify default columns were logged
        params_call = mock_mlflow.log_params.call_args[0][0]
        assert "feature_columns" in params_call
        logged_features = json.loads(params_call["feature_columns"])
        assert logged_features == DataLoader.FEATURE_COLUMNS
