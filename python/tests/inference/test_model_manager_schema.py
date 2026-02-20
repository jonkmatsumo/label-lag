import hashlib
import json
from unittest.mock import MagicMock, patch

from forecast.model_manager import ModelManager


class TestModelManagerSchemaHash:
    def _fresh_manager(self):
        ModelManager._instance = None
        return ModelManager()

    def test_schema_mismatch_detected(self):
        manager = self._fresh_manager()

        # Features in inference
        required_features = ["f1", "f2"]

        # Stored hash info with DIFFERENT features
        stored_features = ["f1", "f2", "f3"]
        stored_json = json.dumps(sorted(stored_features))
        stored_hash = hashlib.sha256(stored_json.encode("utf-8")).hexdigest()
        stored_hash_info = {"feature_schema_hash": stored_hash, "feature_count": 3}

        with (
            patch("mlflow.pyfunc.load_model", return_value=MagicMock()),
            patch("mlflow.MlflowClient"),
            patch.object(manager, "_get_production_version", return_value="1"),
            patch.object(
                manager,
                "_load_required_features_artifact",
                return_value=required_features,
            ),
            patch.object(
                manager,
                "_load_feature_schema_hash_artifact",
                return_value=stored_hash_info,
            ),
            patch.object(
                manager, "_load_calibrator_artifact", return_value=(None, False)
            ),
            patch.object(
                manager, "_load_baseline_distribution_artifact", return_value=None
            ),
            patch.object(manager, "_benchmark_inference"),
        ):
            assert manager.load_production_model() is True
            assert manager.schema_mismatch_detected is True

    def test_schema_match_succeeds(self):
        manager = self._fresh_manager()

        required_features = ["f1", "f2"]
        ordered_features = sorted(required_features)
        expected_json = json.dumps(ordered_features)
        expected_hash = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()

        stored_hash_info = {"feature_schema_hash": expected_hash, "feature_count": 2}

        with (
            patch("mlflow.pyfunc.load_model", return_value=MagicMock()),
            patch("mlflow.MlflowClient"),
            patch.object(manager, "_get_production_version", return_value="1"),
            patch.object(
                manager,
                "_load_required_features_artifact",
                return_value=required_features,
            ),
            patch.object(
                manager,
                "_load_feature_schema_hash_artifact",
                return_value=stored_hash_info,
            ),
            patch.object(
                manager, "_load_calibrator_artifact", return_value=(None, False)
            ),
            patch.object(
                manager, "_load_baseline_distribution_artifact", return_value=None
            ),
            patch.object(manager, "_benchmark_inference"),
        ):
            assert manager.load_production_model() is True
            assert manager.schema_mismatch_detected is False
