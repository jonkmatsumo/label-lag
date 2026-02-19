"""Regression tests for disjoint tuning vs early-stopping validation slices."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from model.loader import TrainTestSplit
from model.train import train_model
from training.schemas import TuningConfig


@patch("features.registry.FeatureRegistry.get")
def test_tuning_and_early_stopping_use_disjoint_validation_sets(_mock_registry):
    """Tune validation and early-stop validation must not overlap."""
    n_train = 100
    x_train = pd.DataFrame(
        {
            "a": np.arange(n_train),
            "b": np.arange(n_train, dtype=float),
            "c": np.linspace(0.0, 1.0, n_train),
        }
    )
    y_train = pd.Series(([0, 1] * (n_train // 2))[:n_train])

    x_test = pd.DataFrame(
        {"a": [201, 202, 203], "b": [1.0, 2.0, 3.0], "c": [0.2, 0.4, 0.6]}
    )
    y_test = pd.Series([0, 1, 0])

    split = TrainTestSplit(
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        y_test=y_test,
    )

    captured: dict[str, list[int] | None] = {
        "tune_val_idx": None,
        "es_val_idx": None,
    }

    class DummyXGBClassifier:
        def __init__(self, *args, **kwargs):
            self.best_iteration = 3
            self.feature_importances_ = np.array([0.5, 0.3, 0.2])

        def fit(self, x_fit, y_fit, eval_set=None):
            if eval_set:
                captured["es_val_idx"] = list(eval_set[0][0].index)
            return self

        def predict(self, x):
            return np.zeros(len(x), dtype=int)

        def predict_proba(self, x):
            probs = np.linspace(0.1, 0.9, len(x))
            return np.column_stack([1.0 - probs, probs])

    def _fake_run_tuning_study(x_tr, y_tr, x_val, y_val, **kwargs):
        captured["tune_val_idx"] = list(x_val.index)
        trials = pd.DataFrame(
            {
                "trial": [0],
                "state": ["TrialState.COMPLETE"],
                "value": [0.5],
            }
        )
        return {}, trials

    with (
        patch("model.train._get_mlflow") as mock_get_mlflow,
        patch("model.train.DataLoader") as mock_loader_cls,
        patch("model.train.XGBClassifier", DummyXGBClassifier),
        patch("model.train.run_tuning_study", _fake_run_tuning_study),
        patch("mlflow.models.infer_signature"),
        patch("joblib.dump"),
    ):
        mock_mlflow = MagicMock()
        mock_get_mlflow.return_value = mock_mlflow
        mock_run = MagicMock()
        mock_run.info.run_id = "run_456"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = ["a", "b", "c"]
        mock_loader.load_train_test_split.return_value = split
        mock_loader_cls.return_value = mock_loader

        train_model(
            feature_columns=["a", "b", "c"],
            tuning_config=TuningConfig(enabled=True, n_trials=5),
            early_stopping_rounds=10,
        )

    tune_idx = captured["tune_val_idx"]
    es_idx = captured["es_val_idx"]

    assert tune_idx is not None
    assert es_idx is not None
    assert set(tune_idx).isdisjoint(set(es_idx))

    # Deterministic expected split with defaults:
    # train(100) -> calibration tail(20) => fit_base(80)
    # tuning val fraction(0.2): val_full indices [64..79], split in half
    assert tune_idx == list(range(64, 72))
    assert es_idx == list(range(72, 80))
