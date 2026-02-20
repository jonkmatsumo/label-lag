import threading
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from forecast.model_manager import ModelManager, ModelStateBundle


class TestAdversarialConcurrency:
    def _fresh_manager(self):
        ModelManager._instance = None
        m = ModelManager()
        return m

    def test_predict_during_reload_atomic_swap(self):
        """Verify that predict works while a reload is in progress (using old model)."""
        manager = self._fresh_manager()

        # 1. Setup initial model
        old_model = MagicMock()
        old_model.predict.return_value = [0.1]
        old_bundle = ModelStateBundle(
            model=old_model,
            version="v1",
            source="mlflow",
            required_features=["f1"],
            calibrator=None,
            calibrator_loaded=False,
            baseline_distribution=None,
            feature_importance=None,
            last_reload_ts=time.time(),
        )
        manager._bundle = old_bundle
        manager._state = "ready"

        # 2. Setup new model load that blocks
        new_model = MagicMock()
        new_model.predict.return_value = [0.9]
        new_bundle = ModelStateBundle(
            model=new_model,
            version="v2",
            source="mlflow",
            required_features=["f1"],
            calibrator=None,
            calibrator_loaded=False,
            baseline_distribution=None,
            feature_importance=None,
            last_reload_ts=time.time(),
        )

        reload_started = threading.Event()
        reload_can_finish = threading.Event()

        def mocked_load_mlflow():
            reload_started.set()
            reload_can_finish.wait(timeout=5)
            return new_bundle

        # 3. Trigger reload in background
        with (
            patch.object(manager, "_load_from_mlflow", side_effect=mocked_load_mlflow),
            patch.object(manager, "_benchmark_inference"),
        ):
            reload_thread = threading.Thread(target=manager.load_production_model)
            reload_thread.start()

            # Wait for reload to start
            assert reload_started.wait(timeout=2)
            # During atomic swap, we stay in 'ready' to allow serving from old model
            assert manager._state == "ready"

            # 4. Attempt predict during "loading" state
            # Should use OLD bundle because it's an atomic swap
            df = pd.DataFrame({"f1": [1.0]})
            pred = manager.predict(df)

            assert pred == [0.1]
            old_model.predict.assert_called_once()
            new_model.predict.assert_not_called()

            # 5. Finish reload
            reload_can_finish.set()
            reload_thread.join(timeout=5)

            assert manager._state == "ready"
            assert manager.model_version == "v2"

            # 6. Predict after reload uses NEW model
            pred_new = manager.predict(df)
            assert pred_new == [0.9]
            new_model.predict.assert_called_once()

    def test_predict_fails_if_only_loading_and_no_bundle(self):
        """Verify that predict raises error if loading and no bundle exists yet."""
        manager = self._fresh_manager()
        manager._bundle = None
        manager._state = "loading"

        df = pd.DataFrame({"f1": [1.0]})
        with pytest.raises(RuntimeError) as excinfo:
            manager.predict(df)

        assert "Model reload in progress" in str(excinfo.value)
        assert excinfo.value.args[1] == "reload_in_progress"

    def test_benchmark_mlflow_write_failure(self):
        """Verify that MLflow write failure during benchmark does not crash reload."""
        manager = self._fresh_manager()
        mock_bundle = ModelStateBundle(
            model=MagicMock(),
            version="v1",
            source="mlflow",
            required_features=[],
            calibrator=None,
            calibrator_loaded=False,
            baseline_distribution=None,
            feature_importance=None,
            last_reload_ts=time.time(),
        )

        with (
            patch.object(manager, "_load_from_mlflow", return_value=mock_bundle),
            patch("mlflow.MlflowClient") as mock_client_cls,
        ):
            mock_client = mock_client_cls.return_value
            # Simulate failure during log_metric
            mock_client.log_metric.side_effect = Exception("S3 Upload Failed")

            # This should NOT raise Exception to the caller
            res = manager.load_production_model()
            assert res is True
            assert manager._state == "ready"
            # It still finished benchmarking even if MLflow log failed
            assert manager._benchmark_last_status == "success"
