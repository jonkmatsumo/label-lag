"""Tests for strict-mode visibility in forecast health payloads."""

from types import SimpleNamespace
from unittest.mock import patch

from forecast.service import ForecastService


def _manager_with_config(config: dict[str, bool]):
    return SimpleNamespace(
        model_version="v1",
        model_loaded=True,
        get_diagnostics=lambda: {"config": config},
    )


def test_health_components_default_strict_flags_false():
    manager = _manager_with_config(
        {
            "strict_feature_schema": False,
            "strict_tuning_resume_validation": False,
            "strict_split_strategy_validation": False,
        }
    )
    forecaster = SimpleNamespace(model_version="unknown")

    with (
        patch("forecast.service.get_model_manager", return_value=manager),
        patch("forecast.service.get_forecaster", return_value=forecaster),
    ):
        response = ForecastService().GetHealth(request=None, context=None)

    assert response.components["strict_feature_schema"] == "false"
    assert response.components["strict_tuning_resume_validation"] == "false"
    assert response.components["strict_split_strategy_validation"] == "false"


def test_health_components_reflect_enabled_strict_flags():
    manager = _manager_with_config(
        {
            "strict_feature_schema": True,
            "strict_tuning_resume_validation": True,
            "strict_split_strategy_validation": True,
        }
    )
    forecaster = SimpleNamespace(model_version="unknown")

    with (
        patch("forecast.service.get_model_manager", return_value=manager),
        patch("forecast.service.get_forecaster", return_value=forecaster),
    ):
        response = ForecastService().GetHealth(request=None, context=None)

    assert response.components["strict_feature_schema"] == "true"
    assert response.components["strict_tuning_resume_validation"] == "true"
    assert response.components["strict_split_strategy_validation"] == "true"
