"""Runtime strict-mode config discoverability in diagnostics payloads."""

from forecast.model_manager import ModelManager


def _fresh_manager() -> ModelManager:
    ModelManager._instance = None
    return ModelManager()


def test_diagnostics_config_defaults_to_all_strict_flags_off(monkeypatch):
    monkeypatch.delenv("ENFORCE_MODEL_FEATURES", raising=False)
    monkeypatch.delenv("STRICT_TUNING_RESUME_VALIDATION", raising=False)
    monkeypatch.delenv("STRICT_SPLIT_STRATEGY_VALIDATION", raising=False)

    manager = _fresh_manager()
    diagnostics = manager.get_diagnostics()

    expected = {
        "strict_feature_schema": False,
        "strict_tuning_resume_validation": False,
        "strict_split_strategy_validation": False,
    }
    assert diagnostics["config"] == expected
    assert diagnostics["ml_health"]["config"] == expected
    assert manager.get_ml_health_summary()["config"] == expected
    assert set(diagnostics["config"].keys()) == set(expected.keys())
    assert all(isinstance(value, bool) for value in diagnostics["config"].values())


def test_diagnostics_config_reflects_enabled_strict_flags(monkeypatch):
    monkeypatch.setenv("ENFORCE_MODEL_FEATURES", "true")
    monkeypatch.setenv("STRICT_TUNING_RESUME_VALIDATION", "1")
    monkeypatch.setenv("STRICT_SPLIT_STRATEGY_VALIDATION", "yes")

    manager = _fresh_manager()
    diagnostics = manager.get_diagnostics()

    expected = {
        "strict_feature_schema": True,
        "strict_tuning_resume_validation": True,
        "strict_split_strategy_validation": True,
    }
    assert diagnostics["config"] == expected
    assert diagnostics["ml_health"]["config"] == expected
    assert manager.get_ml_health_summary()["config"] == expected
    assert set(diagnostics["config"].keys()) == set(expected.keys())
    assert all(isinstance(value, bool) for value in diagnostics["config"].values())


def test_diagnostics_config_parses_false_like_env_values(monkeypatch):
    monkeypatch.setenv("ENFORCE_MODEL_FEATURES", "false")
    monkeypatch.setenv("STRICT_TUNING_RESUME_VALIDATION", "0")
    monkeypatch.setenv("STRICT_SPLIT_STRATEGY_VALIDATION", "off")

    manager = _fresh_manager()
    diagnostics = manager.get_diagnostics()

    expected = {
        "strict_feature_schema": False,
        "strict_tuning_resume_validation": False,
        "strict_split_strategy_validation": False,
    }
    assert diagnostics["config"] == expected
    assert diagnostics["ml_health"]["config"] == expected
    assert manager.get_ml_health_summary()["config"] == expected
