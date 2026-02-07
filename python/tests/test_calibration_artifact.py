"""Tests for calibration artifact persistence and loading."""

from unittest.mock import MagicMock, patch

import numpy as np

from forecast_server.model_manager import ModelManager
from model.evaluate import ScoreCalibrator
from model.train import train_model


class TestCalibrationArtifact:
    """Tests for calibration artifact flow (C2)."""

    def test_calibrator_fitting_during_training(self):
        """Test that ScoreCalibrator is fitted and logged during training (C2)."""
        # Mock mlflow
        with (
            patch("model.train._get_mlflow") as mock_get_mlflow,
            patch("model.train.DataLoader") as mock_loader,
            patch("model.train.XGBClassifier") as mock_xgb,
            patch("mlflow.models.infer_signature"),
            patch("joblib.dump") as mock_joblib_dump,
        ):
            mock_mlflow = MagicMock()
            mock_get_mlflow.return_value = mock_mlflow

            # Mock data loader split
            mock_loader.return_value.FEATURE_COLUMNS = ["feat1", "feat2"]
            mock_split = MagicMock()
            mock_split.train_size = 100
            mock_split.test_size = 50
            mock_split.X_train = MagicMock()
            mock_split.X_test = MagicMock()
            mock_split.y_train = np.array([0, 1] * 50)
            mock_split.y_test = np.array([0, 1] * 25)
            mock_split.split_manifest = {}
            mock_loader.return_value.load_train_test_split.return_value = mock_split

            # Mock XGBoost
            mock_clf = MagicMock()
            mock_xgb.return_value = mock_clf
            mock_clf.predict.return_value = np.array([0, 1] * 25)
            mock_clf.predict_proba.return_value = np.array(
                [[0.9, 0.1], [0.1, 0.9]] * 25
            )

            # Run training
            train_model(database_url="sqlite:///:memory:")

            # Verify calibrator was logged as artifact
            # We expect mlflow.log_artifact to be called
            # with a path ending in calibrator.pkl.
            artifact_calls = [
                call.args[0] for call in mock_mlflow.log_artifact.call_args_list
            ]
            assert any(path.endswith("calibrator.pkl") for path in artifact_calls)

            # Verify joblib.dump was called
            assert mock_joblib_dump.called

    def test_model_manager_loads_calibrator(self):
        """Test that ModelManager loads calibrator artifact when available (C2)."""
        manager = ModelManager()
        manager._initialized = False  # Force re-init for clean test
        manager.__init__()

        with (
            patch("mlflow.pyfunc.load_model"),
            patch("mlflow.MlflowClient") as mock_client_cls,
            patch("joblib.load") as mock_joblib_load,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            # Mock model version
            mock_version = MagicMock()
            mock_version.current_stage = "Production"
            mock_version.version = "1"
            mock_version.run_id = "run123"
            mock_client.search_model_versions.return_value = [mock_version]

            # Mock artifact download
            mock_client.download_artifacts.side_effect = (
                lambda run_id, path: "/tmp/" + path
            )

            # Mock loaded calibrator
            mock_calibrator = MagicMock()
            mock_joblib_load.return_value = mock_calibrator

            # Load production model
            success = manager.load_production_model()

            assert success is True
            assert manager.calibrator_loaded is True
            assert manager.calibrator == mock_calibrator

    def test_model_manager_fallback_calibrator(self):
        """Test fallback calibrator when artifact is missing (C2)."""
        manager = ModelManager()
        manager._initialized = False
        manager.__init__()

        with (
            patch("mlflow.pyfunc.load_model"),
            patch("mlflow.MlflowClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            # Mock model version
            mock_version = MagicMock()
            mock_version.current_stage = "Production"
            mock_version.version = "1"
            mock_version.run_id = "run123"
            mock_client.search_model_versions.return_value = [mock_version]

            # Mock artifact download failure
            mock_client.download_artifacts.side_effect = Exception("Artifact not found")

            # Load production model
            success = manager.load_production_model()

            assert success is True
            assert manager.calibrator_loaded is False
            assert isinstance(manager.calibrator, ScoreCalibrator)

    def test_forecaster_uses_loaded_calibrator(self):
        """Test that SignalForecaster uses the loaded calibrator and reports it (C2)."""
        from decimal import Decimal

        from forecast_server.services import SignalForecaster
        from training_server.schemas import SignalRequest

        forecaster = SignalForecaster()
        request = SignalRequest(
            user_id="user1", amount=Decimal("100.0"), client_transaction_id="tx123"
        )

        mock_manager = MagicMock()
        mock_manager.model_loaded = True
        mock_manager.model_version = "v1"
        mock_manager.calibrator_loaded = True

        # Mock calibrator to return a specific score
        mock_calibrator = MagicMock()
        mock_calibrator.transform.return_value = np.array([88])
        mock_manager.calibrator = mock_calibrator

        with (
            patch(
                "forecast_server.model_manager.get_model_manager",
                return_value=mock_manager,
            ),
            patch.object(forecaster, "_fetch_features") as mock_fetch,
        ):
            mock_features = MagicMock()
            mock_features.has_history = True
            mock_fetch.return_value = mock_features

            # Mock model prediction
            with patch.object(forecaster, "_predict_with_model", return_value=0.5):
                result = forecaster.predict(request)

                assert result["model_score"] == 88
                assert result["diagnostics"]["calibrator_loaded"] is True
                mock_calibrator.transform.assert_called_once()
