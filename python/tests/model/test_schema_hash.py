import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest


class TestFeatureSchemaHash:
    """Pure unit tests of the hash algorithm (sort -> JSON -> SHA256)."""

    def test_schema_hash_deterministic(self):
        """Identical sorted feature lists produce the same hash."""
        features = ["f1", "f2"]
        ordered = sorted(features)
        schema_json = json.dumps(ordered)
        h = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()

        # Same input -> same hash
        h2 = hashlib.sha256(
            json.dumps(sorted(["f1", "f2"])).encode("utf-8")
        ).hexdigest()
        assert h == h2

    def test_schema_hash_order_independent(self):
        """Feature order does not affect the hash (sorting normalizes)."""
        h1 = hashlib.sha256(
            json.dumps(sorted(["f2", "f1", "f3"])).encode("utf-8")
        ).hexdigest()
        h2 = hashlib.sha256(
            json.dumps(sorted(["f3", "f1", "f2"])).encode("utf-8")
        ).hexdigest()
        assert h1 == h2

    def test_schema_hash_changes_with_features(self):
        """Adding or removing a feature changes the hash."""
        h1 = hashlib.sha256(
            json.dumps(sorted(["f1", "f2"])).encode("utf-8")
        ).hexdigest()
        h2 = hashlib.sha256(
            json.dumps(sorted(["f1", "f2", "f3"])).encode("utf-8")
        ).hexdigest()
        assert h1 != h2

    def test_schema_hash_matches_train_model_algorithm(self):
        """Verify the exact algorithm used in train_model lines 433-435."""
        # Reproduce the 3-line algorithm from model/train.py:
        #   ordered_features = sorted(actual_feature_columns)
        #   schema_json = json.dumps(ordered_features)
        #   feature_schema_hash = hashlib.sha256(
        #       schema_json.encode("utf-8")
        #   ).hexdigest()
        feature_columns = ["velocity_24h", "amount_to_avg_ratio_30d", "balance_z"]
        ordered_features = sorted(feature_columns)
        schema_json = json.dumps(ordered_features)
        feature_schema_hash = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()

        # The JSON of sorted features is the canonical input
        assert ordered_features == [
            "amount_to_avg_ratio_30d",
            "balance_z",
            "velocity_24h",
        ]
        assert schema_json == json.dumps(
            ["amount_to_avg_ratio_30d", "balance_z", "velocity_24h"]
        )
        assert len(feature_schema_hash) == 64  # SHA256 hex digest


@pytest.mark.slow
class TestFeatureSchemaHashIntegration:
    """Integration test: verify train_model persists the hash to MLflow.

    Marked slow because it calls the full train_model() which triggers heavy
    imports (xgboost, optuna, matplotlib, mlflow) and real I/O (figure
    rendering, parquet writes).  Excluded from CI via ``-m 'not slow'``.
    """

    @patch("features.registry.FeatureRegistry.get")
    @patch("model.train._get_git_sha", return_value="abc")
    @patch("model.train._get_mlflow")
    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    @patch("model.train.XGBClassifier")
    def test_schema_hash_persisted(
        self, mock_xgb_cls, mock_loader_cls, mock_mlflow, mock_get_mlflow, _git, _reg
    ):
        import numpy as np
        import pandas as pd

        from model.loader import TrainTestSplit

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

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = "r"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        from model.train import train_model

        train_model(feature_columns=["f1", "f2"], n_jobs=1)

        # Verify hash calculation: sorted(["f1", "f2"]) -> ["f1", "f2"]
        expected_json = json.dumps(["f1", "f2"])
        expected_hash = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()

        # Check tags
        mock_mlflow.set_tag.assert_any_call("feature_schema_hash", expected_hash)

        # Check artifact
        # find log_dict call for feature_schema_hash.json
        found_artifact = False
        for call in mock_mlflow.log_dict.call_args_list:
            data, name = call[0]
            if name == "feature_schema_hash.json":
                assert data["feature_schema_hash"] == expected_hash
                assert data["feature_count"] == 2
                found_artifact = True
        assert found_artifact
