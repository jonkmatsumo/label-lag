"""Unit tests for Optuna tuning module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from model.tuning import (
    DEFAULT_SEARCH_SPACE,
    get_trial_params,
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


class TestGetTrialParams:
    """get_trial_params extracts specific trial hyperparameters."""

    def test_get_trial_params_returns_specific_trial(self):
        """Extracts params for specified trial number."""
        trials_df = pd.DataFrame(
            {
                "trial": [0, 1, 2],
                "value": [0.7, 0.8, 0.75],
                "state": ["COMPLETE", "COMPLETE", "COMPLETE"],
                "params_max_depth": [6, 8, 7],
                "params_learning_rate": [0.1, 0.15, 0.12],
            }
        )

        params = get_trial_params(trials_df, 1)
        assert params["max_depth"] == 8
        assert params["learning_rate"] == 0.15

    def test_get_trial_params_returns_empty_when_not_found(self):
        """Returns empty dict when trial number doesn't exist."""
        trials_df = pd.DataFrame(
            {
                "trial": [0, 1],
                "value": [0.7, 0.8],
                "params_max_depth": [6, 8],
            }
        )

        params = get_trial_params(trials_df, 99)
        assert params == {}

    def test_get_trial_params_raises_on_negative(self):
        """Raises ValueError for negative trial number."""
        trials_df = pd.DataFrame({"trial": [0], "value": [0.7]})
        with pytest.raises(ValueError, match="non-negative"):
            get_trial_params(trials_df, -1)


class TestSelectedTrialOverride:
    """Manual trial selection overrides best trial."""

    @patch("model.train._get_git_sha", return_value="abc")
    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    def test_selected_trial_overrides_best(
        self, mock_loader_cls, mock_mlflow, _mock_git
    ):
        """When selected_trial_number is set, uses that trial's params."""
        from api.schemas import TuningConfig
        from model.loader import TrainTestSplit
        from model.train import train_model

        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = ["a", "b", "c"]
        mock_loader_cls.return_value = mock_loader

        n_train = 50
        x_train = pd.DataFrame(
            {
                "a": np.random.rand(n_train),
                "b": np.random.rand(n_train),
                "c": np.random.rand(n_train),
            }
        )
        y_train = pd.Series(np.random.randint(0, 2, n_train))
        y_train.iloc[:5] = 1

        mock_split = TrainTestSplit(
            X_train=x_train,
            y_train=y_train,
            X_test=pd.DataFrame({"a": [1], "b": [1], "c": [1]}),
            y_test=pd.Series([0]),
        )
        mock_loader.load_train_test_split.return_value = mock_split

        mock_run = MagicMock()
        mock_run.info.run_id = "run_xyz"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.set_experiment.return_value = None

        tuning_config = TuningConfig(enabled=True, n_trials=5, selected_trial_number=2)

        with patch("model.train.run_tuning_study") as mock_tune:
            # Mock tuning to return specific trial params
            mock_trials_df = pd.DataFrame(
                {
                    "trial": [0, 1, 2],
                    "value": [0.7, 0.8, 0.75],
                    "state": ["COMPLETE", "COMPLETE", "COMPLETE"],
                    "params_max_depth": [6, 8, 10],
                    "params_learning_rate": [0.1, 0.15, 0.2],
                }
            )
            mock_tune.return_value = (
                {"max_depth": 8, "learning_rate": 0.15},
                mock_trials_df,
            )

            train_model(
                tuning_config=tuning_config,
                feature_columns=["a", "b", "c"],
            )

            # Verify set_tags was called with manual selection
            assert mock_mlflow.set_tags.called
            all_calls = mock_mlflow.set_tags.call_args_list
            tuning_tags_found = False
            for call in all_calls:
                tags = call[0][0] if call[0] else {}
                if "tuning.selected_trial" in tags:
                    assert tags["tuning.selected_trial"] == "2"
                    assert tags["tuning.selection_type"] == "manual"
                    tuning_tags_found = True
                    break
            assert tuning_tags_found, "Tuning tags not found"
