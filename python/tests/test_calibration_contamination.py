"""Regression tests for calibration split contamination hardening."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from model.loader import TrainTestSplit
from model.train import train_model


@patch("features.registry.FeatureRegistry.get")
def test_calibrator_fit_uses_calibration_labels_only(_mock_registry):
    """Calibrator must fit on train-derived calibration labels, never y_test."""
    x_train = pd.DataFrame(
        {
            "a": list(range(10)),
            "b": [float(i) for i in range(10)],
            "c": [i / 10.0 for i in range(10)],
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    x_test = pd.DataFrame(
        {
            "a": [100, 101, 102],
            "b": [1.0, 2.0, 3.0],
            "c": [0.2, 0.4, 0.6],
        }
    )
    y_test = pd.Series([1, 1, 1])

    split = TrainTestSplit(
        X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test
    )

    expected_y_cal = y_train.iloc[-2:].to_numpy()
    calibration_probs = np.array([0.2, 0.8])
    test_probs = np.array([0.1, 0.4, 0.9])

    captured: dict[str, np.ndarray | list[np.ndarray] | None] = {
        "fit_labels": None,
        "fit_probs": None,
        "transform_probs": [],
    }

    class CapturingCalibrator:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, y_prob, y_true):
            captured["fit_probs"] = np.asarray(y_prob)
            captured["fit_labels"] = np.asarray(y_true)
            return self

        def transform(self, y_prob):
            arr = np.asarray(y_prob)
            captured["transform_probs"].append(arr)
            return np.clip((arr * 98 + 1).astype(int), 1, 99)

    def _predict_proba_side_effect(features):
        rows = len(features)
        if rows == len(calibration_probs):
            probs = calibration_probs
        elif rows == len(test_probs):
            probs = test_probs
        else:
            probs = np.full(rows, 0.5)
        return np.column_stack([1 - probs, probs])

    with (
        patch("model.train._get_mlflow") as mock_get_mlflow,
        patch("model.train.DataLoader") as mock_loader_cls,
        patch("model.train.XGBClassifier") as mock_xgb_cls,
        patch("model.evaluate.ScoreCalibrator", CapturingCalibrator),
        patch("mlflow.models.infer_signature"),
        patch("joblib.dump"),
    ):
        mock_mlflow = MagicMock()
        mock_get_mlflow.return_value = mock_mlflow
        mock_run = MagicMock()
        mock_run.info.run_id = "run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = ["a", "b", "c"]
        mock_loader.load_train_test_split.return_value = split
        mock_loader_cls.return_value = mock_loader

        mock_clf = MagicMock()
        mock_clf.predict.return_value = np.array([0, 0, 1])
        mock_clf.predict_proba.side_effect = _predict_proba_side_effect
        mock_clf.feature_importances_ = np.array([0.6, 0.3, 0.1])
        mock_xgb_cls.return_value = mock_clf

        train_model(feature_columns=["a", "b", "c"])

    assert np.array_equal(captured["fit_labels"], expected_y_cal)
    assert np.array_equal(captured["fit_probs"], calibration_probs)
    assert not np.array_equal(captured["fit_labels"], y_test.to_numpy())
    assert any(np.array_equal(p, test_probs) for p in captured["transform_probs"])

    params_logged = mock_mlflow.log_params.call_args[0][0]
    assert params_logged["calibration_set_size"] == 2
    assert params_logged["calibration_positive_rate"] == 0.5
    assert params_logged["calibration_split_strategy"] == "train_tail_fraction"
