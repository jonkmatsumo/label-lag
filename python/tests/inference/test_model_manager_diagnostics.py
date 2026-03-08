import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from forecast.model_manager import ModelManager
from training.reason_codes import OPERABILITY_STATUSES


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
        # MLflow fails with "artifact" in error to trigger "artifact_missing" reason
        with (
            patch.object(
                manager,
                "_load_from_mlflow",
                side_effect=lambda: setattr(
                    manager, "_mlflow_failure_reason", "artifact_missing"
                )
                or None,
            ),
            patch.object(manager, "_load_fallback_model", return_value=None),
            patch("forecast.metrics.model_reload_failure_total.labels") as mock_labels,
        ):
            mock_inc = mock_labels.return_value.inc
            manager.load_production_model()
            diag = manager.get_diagnostics()

            assert diag["state"] == "failed"
            assert diag["last_error"] == "Both MLflow and fallback failed"
            assert diag["last_reload_status"] == "failed"
            assert diag["active_model_version"] == "unknown"

            # Check that metrics was called with correct label
            mock_labels.assert_called_with(reason="artifact_missing")
            assert mock_inc.called

    def test_diagnostics_benchmark_status(self):
        from forecast.model_manager import ModelStateBundle

        manager = self._fresh_manager()

        # Use real bundle object for better validation
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
            patch.object(manager, "_benchmark_inference") as mock_benchmark,
        ):
            # 1. Success path: instance-level mock doesn't get 'self'
            def mock_success(*args, **kwargs):
                manager._benchmark_last_status = "success"
                manager._benchmark_last_run_ts = 1234.5

            mock_benchmark.side_effect = mock_success

            res = manager.load_production_model()
            assert res is True
            assert mock_benchmark.called

            diag = manager.get_diagnostics()
            assert diag["benchmark_last_status"] == "success"
            assert diag["benchmark_last_run_ts"] == 1234.5

            # 2. Failure path
            def mock_failure(*args, **kwargs):
                manager._benchmark_last_status = "failed"
                manager._benchmark_last_run_ts = 1235.5

            mock_benchmark.side_effect = mock_failure
            # Reset manager state to allow reload re-triggering benchmark
            manager._benchmarked_versions = set()
            manager.load_production_model()
            diag = manager.get_diagnostics()
            assert diag["benchmark_last_status"] == "failed"
            assert diag["benchmark_last_run_ts"] == 1235.5

    def test_diagnostics_track_feature_coverage_warning_state(self):
        manager = self._fresh_manager()

        initial = manager.get_diagnostics()
        assert initial["feature_coverage_warning_active"] is False
        assert initial["feature_coverage_warning_last_seen_ts"] is None

        manager.update_feature_coverage_warning(active=True, observed_ts=111.5)
        warned = manager.get_diagnostics()
        assert warned["feature_coverage_warning_active"] is True
        assert warned["feature_coverage_warning_last_seen_ts"] == 111.5

        manager.update_feature_coverage_warning(active=False, observed_ts=222.0)
        recovered = manager.get_diagnostics()
        assert recovered["feature_coverage_warning_active"] is False
        # Last warning timestamp remains set to latest warning event.
        assert recovered["feature_coverage_warning_last_seen_ts"] == 111.5

    def test_diagnostics_track_feature_coverage_last_ratio(self):
        manager = self._fresh_manager()

        initial = manager.get_diagnostics()
        assert initial["feature_coverage_last_ratio"] is None
        assert initial["ml_health"]["feature_coverage"]["last_ratio"] is None

        manager.update_feature_coverage_warning(
            active=True, observed_ts=111.5, ratio=0.42
        )
        warned = manager.get_diagnostics()
        assert warned["feature_coverage_last_ratio"] == pytest.approx(0.42)
        assert warned["ml_health"]["feature_coverage"]["last_ratio"] == pytest.approx(
            0.42
        )

        manager.update_feature_coverage_warning(active=False, ratio=3.0)
        recovered = manager.get_diagnostics()
        assert recovered["feature_coverage_last_ratio"] == 1.0
        assert recovered["ml_health"]["feature_coverage"]["last_ratio"] == 1.0

    def test_ml_health_summary_shape_is_stable_and_bounded(self):
        manager = self._fresh_manager()
        manager.update_feature_coverage_warning(
            active=True, observed_ts=111.5, ratio=0.5
        )
        mock_cache = SimpleNamespace(
            _cache=SimpleNamespace(
                computed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                result={
                    "resolution_mode": "stage",
                    "error_code": "no_reference_data",
                    "reference_resolution": {"selected_run_id": "run-1"},
                },
            )
        )

        with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
            health = manager.get_diagnostics()["ml_health"]

        assert {"model", "benchmark", "drift", "feature_coverage"}.issubset(
            health.keys()
        )
        assert isinstance(health["warnings"], list)
        assert health["status"] in OPERABILITY_STATUSES
        assert set(health["model"].keys()) == {
            "state",
            "active_model_version",
            "last_reload_status",
            "last_reload_ts",
            "schema_mismatch_detected",
        }
        assert set(health["benchmark"].keys()) == {
            "enabled",
            "last_status",
            "last_run_ts",
        }
        assert set(health["drift"].keys()) == {
            "reference_resolution_mode",
            "last_error_code",
        }
        assert set(health["feature_coverage"].keys()) == {
            "last_ratio",
            "below_threshold",
        }

        assert isinstance(health["model"]["state"], str)
        assert isinstance(health["model"]["active_model_version"], str)
        assert isinstance(health["model"]["last_reload_status"], str)
        assert health["model"]["last_reload_ts"] is None or isinstance(
            health["model"]["last_reload_ts"], float
        )
        assert isinstance(health["model"]["schema_mismatch_detected"], bool)
        assert isinstance(health["benchmark"]["enabled"], bool)
        assert health["benchmark"]["last_status"] is None or isinstance(
            health["benchmark"]["last_status"], str
        )
        assert health["benchmark"]["last_run_ts"] is None or isinstance(
            health["benchmark"]["last_run_ts"], float
        )
        assert health["drift"]["reference_resolution_mode"] in {
            "alias",
            "stage",
            "latest",
            "none",
        }
        assert health["drift"]["last_error_code"] is None or isinstance(
            health["drift"]["last_error_code"], str
        )
        assert health["feature_coverage"]["last_ratio"] is None or isinstance(
            health["feature_coverage"]["last_ratio"], float
        )
        assert isinstance(health["feature_coverage"]["below_threshold"], bool)

        for value in health.values():
            if isinstance(value, list):
                assert len(value) <= 3

    def test_ml_health_summary_optional_fields_use_null_not_omitted(self):
        manager = self._fresh_manager()
        mock_cache = SimpleNamespace(
            _cache=SimpleNamespace(
                computed_at=None,
                result={"resolution_mode": "none"},
            )
        )

        with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
            health = manager.get_diagnostics()["ml_health"]

        assert set(health.keys()) == {
            "model",
            "benchmark",
            "drift",
            "feature_coverage",
            "config",
            "warnings",
            "status",
            "state",
            "active_model_version",
            "last_reload_status",
            "last_reload_ts",
            "schema_mismatch_detected",
            "benchmark_status",
            "feature_coverage_status",
            "feature_coverage_last_seen_ts",
            "drift_reference_available",
            "drift_resolution_mode",
            "drift_last_computed_ts",
            "drift_last_error_code",
        }
        assert health["benchmark_status"] is None
        assert health["warnings"] == []
        assert health["status"] in OPERABILITY_STATUSES
        assert health["drift"]["last_error_code"] is None
        assert health["drift_last_error_code"] is None
        assert health["drift_last_computed_ts"] is None
        assert health["drift_reference_available"] is False
        assert health["feature_coverage"]["last_ratio"] is None

    def test_ml_health_summary_bounds_drift_strings_and_ignores_dynamic_maps(self):
        manager = self._fresh_manager()
        mock_cache = SimpleNamespace(
            _cache=SimpleNamespace(
                computed_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
                result={
                    "resolution_mode": "x" * 300,
                    "error_code": "e" * 300,
                    "reference_resolution": {
                        "resolution_strategy": {"unexpected": "mapping"},
                        "selected_run_id": "   ",
                        "selected_model_version": "",
                        "unexpected_nested": {"inner": "value"},
                    },
                },
            )
        )

        with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
            health = manager.get_diagnostics()["ml_health"]

        assert set(health["drift"].keys()) == {
            "reference_resolution_mode",
            "last_error_code",
        }
        assert health["drift"]["reference_resolution_mode"] == "none"
        assert health["drift_reference_available"] is False
        assert isinstance(health["drift_last_computed_ts"], float)
        assert len(health["drift"]["last_error_code"]) == 64
        assert len(health["drift_last_error_code"]) == 64
        assert "reference_resolution" not in health["drift"]
