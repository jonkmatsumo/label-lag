"""Guardrails for canonical status and timestamp semantics in diagnostics payloads."""

from types import SimpleNamespace

import pytest

from forecast.model_manager import ModelManager
from training.reason_codes import OPERABILITY_STATUSES, ModelManagerState


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


@pytest.mark.parametrize(
    ("state", "expected_status"),
    [
        (ModelManagerState.IDLE.value, "not_run"),
        (ModelManagerState.LOADING.value, "unknown"),
        (ModelManagerState.READY.value, "success"),
        (ModelManagerState.FAILED.value, "failure"),
    ],
)
def test_diagnostics_and_ml_health_status_are_canonical(state, expected_status):
    manager = _fresh_manager()
    manager._state = state

    diagnostics = manager.get_diagnostics()

    assert diagnostics["status"] == expected_status
    assert diagnostics["status"] in OPERABILITY_STATUSES
    assert diagnostics["ml_health"]["status"] == expected_status


def test_optional_timestamp_fields_use_explicit_nulls_when_invalid():
    manager = _fresh_manager()
    manager._bundle = SimpleNamespace(
        model=object(),
        version="v1",
        source="mlflow",
        required_features=[],
        last_reload_ts="not-a-float",
    )
    manager._benchmark_last_run_ts = "not-a-float"
    manager._feature_coverage_warning_last_seen_ts = "not-a-float"

    diagnostics = manager.get_diagnostics()

    assert diagnostics["last_reload_ts"] is None
    assert diagnostics["benchmark_last_run_ts"] is None
    assert diagnostics["feature_coverage_warning_last_seen_ts"] is None

    health = diagnostics["ml_health"]
    assert "last_reload_ts" in health and health["last_reload_ts"] is None
    assert "feature_coverage_last_seen_ts" in health
    assert health["feature_coverage_last_seen_ts"] is None


def test_unknown_benchmark_status_is_normalized_to_bounded_code():
    manager = _fresh_manager()
    manager._benchmark_last_status = "benchmark failed because dependency x timed out"

    diagnostics = manager.get_diagnostics()

    assert diagnostics["benchmark_last_status"] == "unknown"
    assert diagnostics["benchmark_last_status"] in {
        "skipped_disabled",
        "skipped_sampled_out",
        "success",
        "failed",
        "unknown",
    }
    assert diagnostics["ml_health"]["benchmark_status"] == "unknown"
