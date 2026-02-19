import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from forecast.model_manager import ModelManager, ModelStateBundle


class TestModelManagerConcurrency:
    @pytest.fixture
    def manager(self):
        # ModelManager is a singleton, reset its internal state for tests
        manager = ModelManager()
        with manager._lock:
            manager._bundle = None
            manager._state = "idle"
            manager._last_error = None
        return manager

    def test_predict_during_loading_raises_error(self, manager):
        with manager._lock:
            manager._state = "loading"

        with pytest.raises(RuntimeError) as excinfo:
            manager.predict(pd.DataFrame())
        assert "reload_in_progress" in excinfo.value.args

    def test_predict_serves_old_bundle_during_failed_reload(self, manager):
        old_bundle = ModelStateBundle(
            model=MagicMock(),
            version="v1",
            source="mlflow",
            required_features=[],
            calibrator=MagicMock(),
            calibrator_loaded=True,
            baseline_distribution=None,
            feature_importance=None,
        )
        old_bundle.model.predict.return_value = np.array([0.5])

        with manager._lock:
            manager._bundle = old_bundle
            manager._state = "failed"
            manager._last_error = "reload failed"

        res = manager.predict(pd.DataFrame())
        assert res[0] == 0.5

    def test_atomic_swap_concurrency(self, manager):
        reload_started = threading.Event()
        can_finish_reload = threading.Event()

        # Create two different bundles
        bundle1 = ModelStateBundle(
            model=MagicMock(),
            version="v1",
            source="mlflow",
            required_features=[],
            calibrator=MagicMock(),
            calibrator_loaded=True,
            baseline_distribution=None,
            feature_importance=None,
        )
        bundle1.model.predict.return_value = np.array([0.1])

        bundle2 = ModelStateBundle(
            model=MagicMock(),
            version="v2",
            source="mlflow",
            required_features=[],
            calibrator=MagicMock(),
            calibrator_loaded=True,
            baseline_distribution=None,
            feature_importance=None,
        )
        bundle2.model.predict.return_value = np.array([0.2])

        with manager._lock:
            manager._bundle = bundle1
            manager._state = "ready"

        def mock_load():
            reload_started.set()
            can_finish_reload.wait()  # Block here mid-flight
            return bundle2

        with patch.object(manager, "_load_from_mlflow", side_effect=mock_load):

            def run_reload():
                manager.load_production_model()

            t = threading.Thread(target=run_reload)
            t.start()

            reload_started.wait()
            # At this point, reload is in progress but state is NOT "loading"
            # if we had a bundle because we only transition to loading if
            # not self.model_loaded (atomic swap requirement)

            # Case 1: Prediction during reload should SUCCEED using old bundle
            res = manager.predict(pd.DataFrame())
            assert res[0] == 0.1

            # Allow reload to finish
            can_finish_reload.set()
            t.join()

            # Case 2: After reload, state is "ready" and bundle is bundle2
            assert manager.model_version == "v2"
            res = manager.predict(pd.DataFrame())
            assert res[0] == 0.2

    def test_failed_reload_preserves_old_bundle(self, manager):
        old_bundle = ModelStateBundle(
            model=MagicMock(),
            version="vold",
            source="mlflow",
            required_features=[],
            calibrator=MagicMock(),
            calibrator_loaded=True,
            baseline_distribution=None,
            feature_importance=None,
        )
        old_bundle.model.predict.return_value = np.array([0.8])

        with manager._lock:
            manager._bundle = old_bundle
            manager._state = "ready"

        # Mock MLflow fails, fallback fails
        with patch.object(manager, "_load_from_mlflow", return_value=None):
            with patch.object(manager, "_load_fallback_model", return_value=None):
                success = manager.load_production_model()
                assert success is False
                assert manager._state == "failed"
                assert manager.model_version == "vold"

                # Should still be able to predict from old bundle
                res = manager.predict(pd.DataFrame())
                assert res[0] == 0.8
