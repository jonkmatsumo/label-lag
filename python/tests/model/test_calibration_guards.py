import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from model.loader import TrainTestSplit
from model.train import train_model


class TestCalibrationGuards:
    @pytest.fixture
    def mock_mlflow(self):
        with patch("model.train.mlflow") as m:
            mock_run = MagicMock()
            mock_run.info.run_id = "test_run"
            m.start_run.return_value.__enter__.return_value = mock_run
            # Ensure _get_mlflow() returns this mock
            with patch("model.train._get_mlflow", return_value=m):
                yield m

    @pytest.fixture
    def mock_loader(self):
        with patch("model.train.DataLoader") as m:
            loader = MagicMock()
            loader.FEATURE_COLUMNS = ["f1"]
            m.return_value = loader
            yield loader

    @patch("features.registry.FeatureRegistry.get")
    @patch("model.train._get_git_sha", return_value="abc")
    def test_calibration_skipped_insufficient_samples(
        self, _git, _reg, mock_mlflow, mock_loader, caplog
    ):
        # n=100, frac=0.2 -> cal_size=20
        # Ensure fit set (first 80) is diverse
        y = pd.Series([0, 1] + [0] * 78 + [0, 1] * 10)
        x_df = pd.DataFrame({"f1": range(len(y))})

        mock_loader.load_train_test_split.return_value = TrainTestSplit(
            X_train=x_df, y_train=y, X_test=x_df[:2], y_test=y[:2]
        )

        with caplog.at_level(logging.WARNING):
            with patch("model.evaluate.ScoreCalibrator.fit") as mock_fit:
                # thresholds: n=150 (cal has 20)
                train_model(min_cal_samples=150, n_jobs=1)
                assert not mock_fit.called

            assert (
                "Skipping calibration reason_code="
                "calibration_skipped_insufficient_samples" in caplog.text
            )

        logged_params = {}
        for call in mock_mlflow.log_params.call_args_list:
            logged_params.update(call[0][0])

        assert logged_params["calibration_enabled"] is False
        assert (
            logged_params["calibration_skip_reason"]
            == "calibration_skipped_insufficient_samples"
        )

    @patch("features.registry.FeatureRegistry.get")
    @patch("model.train._get_git_sha", return_value="abc")
    def test_calibration_skipped_insufficient_positives(
        self, _git, _reg, mock_mlflow, mock_loader, caplog
    ):
        # n=100, frac=0.2 -> cal_size=20
        # Ensure fit set (first 80) is diverse
        y = pd.Series(
            [0, 1] + [0] * 78 + [0] * 18 + [1] * 2
        )  # 2 positives in tail (cal set)
        x_df = pd.DataFrame({"f1": range(len(y))})

        mock_loader.load_train_test_split.return_value = TrainTestSplit(
            X_train=x_df, y_train=y, X_test=x_df[:2], y_test=y[:2]
        )

        with caplog.at_level(logging.WARNING):
            with patch("model.evaluate.ScoreCalibrator.fit") as mock_fit:
                train_model(min_cal_samples=10, min_cal_pos=5, n_jobs=1)
                assert not mock_fit.called

            assert (
                "Skipping calibration reason_code="
                "calibration_skipped_insufficient_positives" in caplog.text
            )

        logged_params = {}
        for call in mock_mlflow.log_params.call_args_list:
            logged_params.update(call[0][0])

        assert logged_params["calibration_enabled"] is False
        assert (
            logged_params["calibration_skip_reason"]
            == "calibration_skipped_insufficient_positives"
        )

    @patch("features.registry.FeatureRegistry.get")
    @patch("model.train._get_git_sha", return_value="abc")
    def test_calibration_skipped_insufficient_negatives(
        self, _git, _reg, mock_mlflow, mock_loader, caplog
    ):
        # n=100, frac=0.2 -> cal_size=20
        y = pd.Series(
            [0, 1] + [0] * 78 + [1] * 18 + [0] * 2
        )  # 2 negatives in tail (cal set)
        x_df = pd.DataFrame({"f1": range(len(y))})

        mock_loader.load_train_test_split.return_value = TrainTestSplit(
            X_train=x_df, y_train=y, X_test=x_df[:2], y_test=y[:2]
        )

        with caplog.at_level(logging.WARNING):
            with patch("model.evaluate.ScoreCalibrator.fit") as mock_fit:
                train_model(min_cal_samples=10, min_cal_neg=5, n_jobs=1)
                assert not mock_fit.called

            assert (
                "Skipping calibration reason_code="
                "calibration_skipped_insufficient_negatives" in caplog.text
            )

        logged_params = {}
        for call in mock_mlflow.log_params.call_args_list:
            logged_params.update(call[0][0])

        assert logged_params["calibration_enabled"] is False
        assert (
            logged_params["calibration_skip_reason"]
            == "calibration_skipped_insufficient_negatives"
        )

    @patch("features.registry.FeatureRegistry.get")
    @patch("model.train._get_git_sha", return_value="abc")
    def test_calibration_success_when_thresholds_met(
        self, _git, _reg, mock_mlflow, mock_loader
    ):
        # n=100, frac=0.2 -> cal_size=20
        y = pd.Series([0, 1] + [0] * 78 + [0, 1] * 10)  # 10 pos, 10 neg in tail
        x_df = pd.DataFrame({"f1": range(len(y))})

        mock_loader.load_train_test_split.return_value = TrainTestSplit(
            X_train=x_df, y_train=y, X_test=x_df[:2], y_test=y[:2]
        )

        with patch("model.evaluate.ScoreCalibrator.fit") as mock_fit:
            train_model(min_cal_samples=20, min_cal_pos=10, min_cal_neg=10, n_jobs=1)
            assert mock_fit.called

        logged_params = {}
        for call in mock_mlflow.log_params.call_args_list:
            logged_params.update(call[0][0])

        assert logged_params["calibration_enabled"] is True
        assert "calibration_skip_reason" not in logged_params
