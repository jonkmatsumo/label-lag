from unittest.mock import MagicMock, patch

from forecast.model_manager import ModelManager


class TestModelManagerDiagnostics:
    def _fresh_manager(self):
        ModelManager._instance = None
        return ModelManager()

    def test_diagnostics_initial_state(self):
        manager = self._fresh_manager()
        diag = manager.get_diagnostics()
        assert diag["state"] == "idle"
        assert diag["model_version"] == "unknown"
        assert diag["has_bundle"] is False

    def test_diagnostics_after_success(self):
        manager = self._fresh_manager()
        with (
            patch("mlflow.pyfunc.load_model", return_value=MagicMock()),
            patch("mlflow.MlflowClient"),
            patch.object(manager, "_get_production_version", return_value="v1.2.3"),
            patch.object(
                manager, "_load_required_features_artifact", return_value=["f1"]
            ),
            patch.object(
                manager, "_load_feature_schema_hash_artifact", return_value=None
            ),
            patch.object(
                manager, "_load_calibrator_artifact", return_value=(None, True)
            ),
            patch.object(
                manager,
                "_load_baseline_distribution_artifact",
                return_value={"0.5": 0.5},
            ),
            patch.object(manager, "_benchmark_inference"),
        ):
            manager.load_production_model()
            diag = manager.get_diagnostics()
            assert diag["state"] == "ready"
            assert diag["model_version"] == "v1.2.3"
            assert diag["model_source"] == "mlflow"
            assert diag["calibrator_loaded"] is True
            assert diag["has_bundle"] is True

    def test_diagnostics_after_failure(self):
        manager = self._fresh_manager()
        # Mock MLflow and Fallback to fail
        with (
            patch.object(manager, "_load_from_mlflow", return_value=None),
            patch.object(manager, "_load_fallback_model", return_value=None),
            patch(
                "forecast.metrics.model_reload_failure_total.inc"
            ) as mock_metrics_inc,
        ):
            manager.load_production_model()
            diag = manager.get_diagnostics()
            assert diag["state"] == "failed"
            assert diag["last_error"] == "Both MLflow and fallback failed"
            assert mock_metrics_inc.called
