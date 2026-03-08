"""Contract guards for bounded operator warning summaries."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from forecast.model_manager import ModelManager
from training.reason_codes import DIAGNOSTICS_WARNING_CODES


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


def test_warnings_field_is_always_present_with_bounded_values():
    manager = _fresh_manager()

    diagnostics = manager.get_diagnostics()

    assert diagnostics["warnings"] == []
    assert diagnostics["ml_health"]["warnings"] == []
    assert set(diagnostics["warnings"]).issubset(DIAGNOSTICS_WARNING_CODES)


def test_warnings_include_expected_codes_without_verbose_strings():
    manager = _fresh_manager()
    manager._schema_mismatch_detected = True
    manager._model_source = "fallback"
    manager.update_feature_coverage_warning(active=True, observed_ts=111.0)

    diagnostics = manager.get_diagnostics()

    assert set(diagnostics["warnings"]).issubset(DIAGNOSTICS_WARNING_CODES)
    assert {
        "schema_mismatch_detected",
        "reload_failed_using_last_known_good",
        "feature_coverage_below_threshold",
    }.issubset(set(diagnostics["warnings"]))
    assert diagnostics["ml_health"]["warnings"] == diagnostics["warnings"]


def test_warnings_capture_drift_reference_unavailable_when_mode_is_none():
    manager = _fresh_manager()
    mock_cache = SimpleNamespace(
        _cache=SimpleNamespace(
            computed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            result={"resolution_mode": "none"},
        )
    )

    with patch("forecast.drift_cache.get_drift_cache", return_value=mock_cache):
        diagnostics = manager.get_diagnostics()

    assert "drift_reference_unavailable" in diagnostics["warnings"]
    assert set(diagnostics["warnings"]).issubset(DIAGNOSTICS_WARNING_CODES)


def test_invalid_warning_strings_are_filtered_from_ml_health_summary():
    manager = _fresh_manager()
    diagnostics = manager.get_diagnostics()
    diagnostics["warnings"] = [
        "freeform operator message with verbose details that should be dropped",
        "feature_coverage_below_threshold",
        "",
        None,
    ]

    health = manager._build_ml_health_summary(diagnostics)

    assert health["warnings"] == ["feature_coverage_below_threshold"]
