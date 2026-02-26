"""Unit tests for Optuna tuning module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from model.tuning import (
    DEFAULT_SEARCH_SPACE,
    get_trial_params,
)


class TestTuningStudy:
    """Tests for run_tuning_study."""

    @patch("features.registry.FeatureRegistry.get")
    def test_disabled_tuning_skipped(self, _mock_registry):
        """When tuning disabled, train_model does not run study."""
        from model.loader import DataLoader, TrainTestSplit
        from model.train import train_model
        from training.schemas import TuningConfig

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

    @patch("features.registry.FeatureRegistry.get")
    @patch("model.train._get_git_sha", return_value="abc")
    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    def test_selected_trial_overrides_best(
        self, mock_loader_cls, mock_mlflow, _mock_git, _mock_registry
    ):
        """When selected_trial_number is set, uses that trial's params."""
        from model.loader import TrainTestSplit
        from model.train import train_model
        from training.schemas import TuningConfig

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


class TestResumeValidation:
    """Tests for study resume validation (invariants)."""

    @patch("model.tuning.optuna.load_study")
    @patch("model.tuning._env_flag")
    def test_resume_invariant_mismatch_actionable_error(
        self, mock_env_flag, mock_load_study
    ):
        """Actionable error when invariants mismatch in strict mode."""
        mock_env_flag.return_value = True  # Strict mode
        mock_study = MagicMock()
        mock_study.user_attrs = {
            "objective_version": "xgb_objective_v2",
            "dataset_identity": "old_data",
            "seed": "42",
        }
        mock_load_study.return_value = mock_study

        with pytest.raises(ValueError) as exc:
            from model.tuning import run_tuning_study

            run_tuning_study(
                pd.DataFrame(),
                pd.Series(),
                pd.DataFrame(),
                pd.Series(),
                dataset_identity="new_data",
                objective_version="xgb_objective_v2",
                job_id="job_123",
                storage_url="sqlite:///test.db",
            )

        assert "optuna_resume_invariant_mismatch_strict" in str(exc.value)
        assert "dataset_identity (expected=new_data, actual=old_data)" in str(exc.value)

    @patch("model.tuning.optuna.load_study")
    @patch("model.tuning.logger")
    def test_resume_legacy_study_warns_once(self, mock_logger, mock_load_study):
        """Legacy studies (missing objective_version) warn once and proceed."""
        mock_study = MagicMock()
        mock_study.user_attrs = {
            "split_config_hash": "some_hash",
            "seed": "42",
        }  # Missing objective_version
        mock_load_study.return_value = mock_study

        with (
            patch("model.tuning.create_tuning_study"),
            patch("model.tuning.DEFAULT_SEARCH_SPACE", {}),
        ):
            # Reset the global warned flag for testing
            import model.tuning
            from model.tuning import run_tuning_study

            model.tuning._LEGACY_STUDY_WARNED = False

            # First run: should warn
            run_tuning_study(
                pd.DataFrame(),
                pd.Series(),
                pd.DataFrame(),
                pd.Series(),
                objective_version="xgb_objective_v2",
                job_id="job_123",
                storage_url="sqlite:///test.db",
                n_trials=0,
            )
            assert any(
                "optuna_resume_legacy_study" in str(call)
                for call in mock_logger.warning.call_args_list
            )

            # Second run: should NOT warn again (process-level rate limit)
            mock_logger.warning.reset_mock()
            run_tuning_study(
                pd.DataFrame(),
                pd.Series(),
                pd.DataFrame(),
                pd.Series(),
                objective_version="xgb_objective_v2",
                job_id="job_456",
                storage_url="sqlite:///test.db",
                n_trials=0,
            )
            assert not any(
                "optuna_resume_legacy_study" in str(call)
                for call in mock_logger.warning.call_args_list
            )
