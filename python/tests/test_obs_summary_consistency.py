"""Operator summary consistency guards for diagnostics payloads."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from forecast.model_manager import ModelManager
from training.reason_codes import ModelManagerState


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


@pytest.mark.parametrize(
    ("state", "expected_status", "expected_degraded"),
    [
        (ModelManagerState.IDLE.value, "not_run", False),
        (ModelManagerState.LOADING.value, "unknown", True),
        (ModelManagerState.READY.value, "success", False),
        (ModelManagerState.FAILED.value, "failure", True),
    ],
)
def test_summary_overall_status_and_degraded_are_deterministic(
    state,
    expected_status,
    expected_degraded,
):
    manager = _fresh_manager()
    manager._state = state

    diagnostics = manager.get_diagnostics()
    health = diagnostics["ml_health"]

    assert health["status"] == expected_status
    assert health["overall_status"] == expected_status
    assert health["degraded"] == expected_degraded
    assert health["degraded"] == (
        bool(diagnostics["degraded_reasons"])
        or health["overall_status"] in {"failure", "unknown"}
    )


def test_summary_warning_flags_match_warnings_payload_exactly():
    manager = _fresh_manager()
    manager._schema_mismatch_detected = True
    manager._model_source = "fallback"
    manager.update_feature_coverage_warning(active=True, observed_ts=111.0, ratio=0.5)
    mock_cache = SimpleNamespace(
        _cache=SimpleNamespace(
            computed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            result={"resolution_mode": "none"},
        )
    )

    with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
        diagnostics = manager.get_diagnostics()

    health = diagnostics["ml_health"]
    assert health["warnings"] == diagnostics["warnings"]
    assert health["has_warnings"] is True
    assert health["warning_count"] == len(health["warnings"])
    assert health["warning_count"] == 4
