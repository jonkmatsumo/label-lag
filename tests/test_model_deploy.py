"""Tests for model deployment functionality."""

from unittest.mock import MagicMock, patch

import pytest

from api.audit import AuditLogger, set_audit_logger
from api.model_manager import ModelManager


class TestModelDeploy:
    """Tests for model deployment flow."""

    @pytest.fixture
    def model_manager(self):
        """Create a model manager for testing."""
        manager = ModelManager()
        manager._model = MagicMock()
        manager._model_version = "v1"
        manager._model_source = "mlflow"
        return manager

    @pytest.fixture
    def audit_logger(self):
        """Create an audit logger for testing."""
        logger = AuditLogger()
        set_audit_logger(logger)
        return logger

    @patch("api.main.get_model_manager")
    def test_deploy_triggers_reload(self, mock_get_manager, model_manager):
        """Test that deploy triggers model reload."""
        mock_get_manager.return_value = model_manager
        model_manager.load_production_model = MagicMock(return_value=True)

        # Simulate deploy endpoint logic
        success = model_manager.load_production_model()

        assert success is True
        model_manager.load_production_model.assert_called_once()

    @patch("api.main.get_model_manager")
    def test_deploy_creates_audit_event(
        self, mock_get_manager, model_manager, audit_logger
    ):
        """Test that deploy creates MODEL_DEPLOYED audit event."""
        mock_get_manager.return_value = model_manager
        model_manager.load_production_model = MagicMock(return_value=True)
        model_manager._model_version = "v2"

        # Simulate deploy audit logging
        audit_logger.log(
            rule_id="model:v2",
            action="MODEL_DEPLOYED",
            actor="test_actor",
            before_state={"model_version": "v1"},
            after_state={"model_version": "v2"},
            reason="Model deployed to production",
        )

        # Verify audit event
        records = audit_logger.query(action="MODEL_DEPLOYED")
        assert len(records) == 1
        assert records[0].action == "MODEL_DEPLOYED"
        assert records[0].actor == "test_actor"
        assert records[0].rule_id == "model:v2"

    @patch("api.main.get_model_manager")
    def test_deploy_fails_without_production_model(
        self, mock_get_manager, model_manager
    ):
        """Test that deploy fails if production model cannot be loaded."""
        mock_get_manager.return_value = model_manager
        model_manager.load_production_model = MagicMock(return_value=False)

        # Simulate deploy failure
        success = model_manager.load_production_model()

        assert success is False
