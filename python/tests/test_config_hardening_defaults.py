"""Guardrail tests for hardening flag defaults."""

from model.tuning import _env_flag as tuning_env_flag
from training.service import _env_flag as training_env_flag


def test_strict_hardening_flags_default_off_when_unset(monkeypatch):
    monkeypatch.delenv("STRICT_TUNING_RESUME_VALIDATION", raising=False)
    monkeypatch.delenv("STRICT_SPLIT_STRATEGY_VALIDATION", raising=False)

    assert tuning_env_flag("STRICT_TUNING_RESUME_VALIDATION", default=False) is False
    assert training_env_flag("STRICT_SPLIT_STRATEGY_VALIDATION", default=False) is False
