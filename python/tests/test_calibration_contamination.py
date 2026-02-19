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


@patch("features.registry.FeatureRegistry.get")
def test_calibration_edge_cases_tiny_data(_mock_registry):
    """Ensure training completes and fallback strategy is logged for tiny datasets."""
    # n=2, 20% fraction = 0 desired (int(2*0.2)=0).
    # desired_cal_size = max(1, 0) = 1.
    # cal_size = min(1, max(0, 2-2)) = 0.
    x_train = pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]})
    y_train = pd.Series([0, 1])
    x_test = pd.DataFrame({"a": [5], "b": [5.0]})
    y_test = pd.Series([1])

    split = TrainTestSplit(
        X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test
    )

    with (
        patch("model.train._get_mlflow") as mock_get_mlflow,
        patch("model.train.DataLoader") as mock_loader_cls,
        patch("model.train.XGBClassifier") as mock_xgb_cls,
        patch("mlflow.models.infer_signature"),
        patch("joblib.dump"),
    ):
        mock_mlflow = MagicMock()
        mock_get_mlflow.return_value = mock_mlflow
        mock_run = MagicMock()
        mock_run.info.run_id = "run_tiny"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = ["a", "b"]
        mock_loader.load_train_test_split.return_value = split
        mock_loader_cls.return_value = mock_loader

        mock_clf = MagicMock()
        mock_clf.predict.side_effect = lambda x: np.zeros(len(x))
        mock_clf.predict_proba.side_effect = lambda x: np.array([[0.5, 0.5]] * len(x))
        mock_clf.feature_importances_ = np.array([0.5, 0.5])
        mock_xgb_cls.return_value = mock_clf

        train_model(feature_columns=["a", "b"])

    params_logged = mock_mlflow.log_params.call_args[0][0]
    # Default validation_fraction is 0.2. 20% of 2 is 0.4 -> 0.
    # It falls back to no calibration, and then the final strategy is overridden to
    # 'train_tail_fraction_fallback_full_train' because calibration_set_size == 0.
    assert params_logged["calibration_set_size"] == 0
    assert (
        params_logged["calibration_split_strategy"]
        == "train_tail_fraction_fallback_full_train"
    )
    assert params_logged["train_fit_set_size"] == 2


@patch("features.registry.FeatureRegistry.get")
def test_calibration_fallback_single_class_fit(_mock_registry):
    """Ensure fallback happens if calibration would leave fit with only one class."""
    # n=10, 40% fraction = 4 desired.
    # Tail 4: [0, 1, 0, 1]. Fit: [0, 0, 0, 0, 0, 0].
    # Fit has only one class. Cal size should be reduced until fit has 2 classes.
    # If classes are [0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
    # Tail 4: [1, 0, 1, 1]. Fit: [0, 0, 0, 0, 0, 1]. Valid.
    # If classes are [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Tail 4: [0, 0, 0, 0]. Fit: [1, 0, 0, 0, 0, 0]. Valid.
    # If classes are [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # Tail 8: Fit [0, 1]. Valid.

    # Dataset where fit would have 1 class if we took 20%
    # n=5, 20% = 1. Fit: [1, 0, 0, 0]. 2 classes.
    pd.Series([1, 0, 0, 0, 0])
    # n=5, 20% = 1. Fit: [0, 1, 1, 1]. 2 classes.
    pd.Series([0, 1, 1, 1, 1])
    # n=5, 20% = 1. Fit: [0, 0, 1, 1]. 2 classes.
    pd.Series([0, 0, 1, 1, 1])
    # n=3, 20% = 0.
    # Let's use n=10, 90% fraction.
    pd.Series([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # split_config.validation_fraction = 0.9. desired_cal_size = 9.
    # cal_size = min(9, 8) = 8.
    # Tail 8: [0, 0, 0, 0, 0, 0, 0, 0]. Fit: [0, 1]. Valid.
    # If y_train was [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # Tail 8: [0, 0, 0, 0, 0, 0, 0, 1]. Fit: [0, 0]. Invalid (1 class).
    # cal_size reduced to 7. Tail 7: [0, 0, 0, 0, 0, 1]. Fit [0, 0, 0]. Invalid.
    # ... cal_size becomes 1. Fit [0, 0, 0, 0, 0, 0, 0, 0, 0]. Invalid.
    # cal_size becomes 0. Strategy: fallback_no_calibration.

    x_train = pd.DataFrame({"a": range(10), "b": [float(i) for i in range(10)]})
    y_train = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    x_test = pd.DataFrame({"a": [10], "b": [10.0]})
    y_test = pd.Series([1])

    split = TrainTestSplit(
        X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test
    )

    from training.schemas import SplitConfig

    with (
        patch("model.train._get_mlflow") as mock_get_mlflow,
        patch("model.train.DataLoader") as mock_loader_cls,
        patch("model.train.XGBClassifier") as mock_xgb_cls,
        patch("mlflow.models.infer_signature"),
        patch("joblib.dump"),
    ):
        mock_mlflow = MagicMock()
        mock_get_mlflow.return_value = mock_mlflow
        mock_run = MagicMock()
        mock_run.info.run_id = "run_fail"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = ["a", "b"]
        mock_loader.load_train_test_split.return_value = split
        mock_loader_cls.return_value = mock_loader

        mock_clf = MagicMock()
        mock_clf.predict.side_effect = lambda x: np.zeros(len(x))
        mock_clf.predict_proba.side_effect = lambda x: np.array([[0.5, 0.5]] * len(x))
        mock_clf.feature_importances_ = np.array([0.5, 0.5])
        mock_xgb_cls.return_value = mock_clf

        # Request 50% calibration to trigger failure to keep 2 classes in fit
        # with a dataset where all but the last sample are class 0.
        train_model(
            feature_columns=["a", "b"],
            split_config=SplitConfig(validation_fraction=0.5),
        )

    params_logged = mock_mlflow.log_params.call_args[0][0]
    assert params_logged["calibration_set_size"] == 0
    assert (
        params_logged["calibration_split_strategy"]
        == "train_tail_fraction_fallback_full_train"
    )
