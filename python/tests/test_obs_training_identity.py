"""Tests for training identity observability artifacts."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch


@patch("features.registry.FeatureRegistry.get")
@patch("model.train._get_git_sha", return_value="abc")
@patch("model.train._get_mlflow")
@patch("model.train.mlflow")
@patch("model.train.DataLoader")
@patch("model.train.XGBClassifier")
def test_training_identity_artifact_emitted(
    mock_xgb_cls, mock_loader_cls, mock_mlflow, mock_get_mlflow, _git, _reg
):
    """Training should emit a compact training_run_identity artifact + tags."""
    import numpy as np
    import pandas as pd

    from model.loader import TrainTestSplit
    from model.train import train_model

    mock_get_mlflow.return_value = mock_mlflow

    mock_loader = MagicMock()
    mock_loader.FEATURE_COLUMNS = ["f1", "f2"]
    mock_loader_cls.return_value = mock_loader

    mock_clf = MagicMock()
    mock_clf.predict.return_value = np.array([0, 1])
    mock_clf.predict_proba.return_value = np.array([[0.9, 0.1], [0.1, 0.9]])
    mock_clf.feature_importances_ = np.array([0.5, 0.5])
    mock_xgb_cls.return_value = mock_clf

    mock_split = TrainTestSplit(
        X_train=pd.DataFrame({"f1": [0, 1], "f2": [0, 1]}),
        y_train=pd.Series([0, 1]),
        X_test=pd.DataFrame({"f1": [0, 1], "f2": [0, 1]}),
        y_test=pd.Series([0, 1]),
    )
    mock_loader.load_train_test_split.return_value = mock_split

    mock_run = MagicMock()
    mock_run.info.run_id = "run-identity-test"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.register_model.return_value = MagicMock(version="9")

    train_model(feature_columns=["f1", "f2"], n_jobs=1)

    expected_hash = hashlib.sha256(json.dumps(["f1", "f2"]).encode("utf-8")).hexdigest()
    mock_mlflow.set_tag.assert_any_call(
        "training_identity.mlflow_run_id", "run-identity-test"
    )
    mock_mlflow.set_tag.assert_any_call(
        "training_identity.feature_schema_hash", expected_hash
    )
    mock_mlflow.set_tag.assert_any_call("training_identity.model_version", "9")

    found_identity_artifact = False
    for call in mock_mlflow.log_dict.call_args_list:
        data, artifact_name = call[0]
        if artifact_name == "training_run_identity.json":
            assert data["schema_version"] == 1
            assert data["mlflow_run_id"] == "run-identity-test"
            assert data["feature_schema_hash"] == expected_hash
            assert data["model_version"] == "9"
            found_identity_artifact = True

    assert found_identity_artifact, "training_run_identity.json was not logged"
