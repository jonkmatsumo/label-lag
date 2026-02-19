from unittest.mock import MagicMock, patch

import pytest

from forecast.model_manager import ModelManager


class TestSafeReload:
    @pytest.fixture
    def manager(self):
        # Reset singleton state
        ModelManager._instance = None
        manager = ModelManager()
        import numpy as np

        # Initialize with a fake model
        manager._bundle = MagicMock()
        manager._bundle.model = MagicMock()
        manager._bundle.model.predict.return_value = np.array([0.1])
        manager._bundle.model.predict_proba.return_value = np.array([[0.9, 0.1]])
        manager._bundle.version = "v1"
        manager._state = "ready"
        manager._initialized = True
        return manager

    def test_reload_failure_preserves_old_model(self, manager):
        """Reload failure should keep old model and set state to failed."""

        # Mock _load_from_mlflow to fail
        with (
            patch.object(manager, "_load_from_mlflow", return_value=None),
            patch.object(manager, "_load_fallback_model", return_value=None),
        ):
            success = manager.load_production_model()

            assert success is False
            assert manager._state == "failed"
            assert manager.model_loaded is True
            assert manager._bundle is not None
            assert manager.model_version == "v1"

            # Predict should still work (with warning logged)
            import pandas as pd

            features = pd.DataFrame({"velocity_24h": [10]})
            prediction = manager.predict(features)
            assert prediction is not None

    def test_reload_success_updates_model(self, manager):
        """Successful reload updates the model."""

        new_bundle = MagicMock()
        new_bundle.version = "v2"
        new_bundle.model = MagicMock()

        with patch.object(manager, "_load_from_mlflow", return_value=new_bundle):
            success = manager.load_production_model()

            assert success is True
            assert manager._state == "ready"
            assert manager.model_loaded is True
            assert manager.model_version == "v2"

    def test_reload_in_progress_serves_old_model(self, manager):
        """During reload, old model is served."""
        # This test ensures we don't transition to loading if we have a model
        # We can't easily simulate concurrency here without threads,
        # but we can check state immediately after calling load_production_model
        # inside a mocked method that hangs?
        # Instead, verify load_production_model logic directly:
        # If we have model, it should NOT set state to "loading"

        # We can mock _transition_to and assert it wasn't called with "loading"
        with (
            patch.object(manager, "_transition_to") as mock_transition,
            patch.object(manager, "_load_from_mlflow", return_value=None),
            patch.object(manager, "_load_fallback_model", return_value=None),
        ):
            manager.load_production_model()

            # Should transition to failed eventually, but NOT to loading first
            # Since we have a model (setup in fixture)

            # Verify calls
            # calls structure: [call("loading"), call("failed", error=...)]
            calls = [c[0][0] for c in mock_transition.call_args_list]
            assert "loading" not in calls
            assert "failed" in calls
