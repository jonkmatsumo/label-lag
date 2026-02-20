import hashlib
import json
from unittest.mock import MagicMock, patch

import pandas as pd

from model.loader import TrainTestSplit
from model.train import train_model


class TestFeatureSchemaHash:
    @patch("features.registry.FeatureRegistry.get")
    @patch("model.train._get_git_sha", return_value="abc")
    @patch("model.train.mlflow")
    @patch("model.train.DataLoader")
    def test_schema_hash_persisted(self, mock_loader_cls, mock_mlflow, _git, _reg):
        mock_loader = MagicMock()
        mock_loader.FEATURE_COLUMNS = ["f1", "f2"]
        mock_loader_cls.return_value = mock_loader

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
