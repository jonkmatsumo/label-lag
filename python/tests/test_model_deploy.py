"""Tests for model deployment functionality."""

from unittest.mock import MagicMock, patch

import pytest

from forecast.model_manager import ModelManager
from forecast.service import ForecastService
from forecast.v1 import forecast_pb2
from training.audit import AuditLogger, set_audit_logger


class TestModelDeploy:
    """Tests for model deployment flow."""

    @pytest.fixture
    def model_manager(self):
        """Create a model manager for testing."""
        from forecast.model_manager import ModelStateBundle

        manager = ModelManager()
        manager._bundle = ModelStateBundle(
            model=MagicMock(),
            version="v1",
            source="mlflow",
            required_features=[],
            calibrator=None,
            calibrator_loaded=False,
            baseline_distribution=None,
            feature_importance=None,
        )
        return manager

    @pytest.fixture
    def audit_logger(self):
        """Create an audit logger for testing."""
        logger = AuditLogger()
        set_audit_logger(logger)
        return logger

    @patch("forecast.service.get_model_manager")
    def test_deploy_triggers_reload(self, mock_get_manager, model_manager):
        """Test that deploy triggers model reload."""
        mock_get_manager.return_value = model_manager
        model_manager.load_production_model = MagicMock(return_value=True)

        service = ForecastService()
        request = forecast_pb2.DeployModelRequest()

        response = service.DeployModel(request, MagicMock())

        assert response.success is True
        model_manager.load_production_model.assert_called_once()

    @patch("forecast.service.get_model_manager")
    def test_deploy_creates_audit_event(
        self, mock_get_manager, model_manager, audit_logger
    ):
        """Test that deploy creates MODEL_DEPLOYED audit event."""
        mock_get_manager.return_value = model_manager
        model_manager.load_production_model = MagicMock(return_value=True)
        model_manager._bundle.version = "v2"

        # Simulate deploy audit logging
        # (happens inside load_production_model or service?)
        # In service.py:
        #   success = manager.load_production_model()
        #   if success: log_audit(...)

        service = ForecastService()
        # ensure actor matches expectations
        request = forecast_pb2.DeployModelRequest(actor="test_actor")

        service.DeployModel(request, MagicMock())

        # Verify audit event
        # Since DeployModel calls audit_logger.log, we don't need to manually call it!
        # Step 545 view showed DeployModel calls audit_logger.log().
        records = audit_logger.query(action="MODEL_DEPLOYED")
        assert len(records) == 1
        assert records[0].action == "MODEL_DEPLOYED"
        assert records[0].actor == "test_actor"
        assert records[0].rule_id == "model:v2"

    @patch("forecast.service.get_model_manager")
    def test_deploy_fails_without_production_model(
        self, mock_get_manager, model_manager
    ):
        """Test that deploy fails if production model cannot be loaded."""
        mock_get_manager.return_value = model_manager
        model_manager.load_production_model = MagicMock(return_value=False)

        service = ForecastService()
        request = forecast_pb2.DeployModelRequest()
        mock_context = MagicMock()
        mock_context.abort.side_effect = Exception("Aborted")

        with pytest.raises(Exception, match="Aborted"):
            service.DeployModel(request, mock_context)

        mock_context.abort.assert_called_once()
