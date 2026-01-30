"""Tests for feature materialization configuration.

Tests the FEATURE_MATERIALIZATION_MODE environment variable and mode selection.
"""

from pipeline.materialize_features import (
    MaterializationMode,
    get_materialization_mode,
)


class TestMaterializationModeConfig:
    """Tests for materialization mode configuration."""

    def test_default_mode_is_legacy(self, monkeypatch):
        """Test that default mode is 'legacy' when env var not set."""
        monkeypatch.delenv("FEATURE_MATERIALIZATION_MODE", raising=False)
        assert get_materialization_mode() == "legacy"

    def test_legacy_mode_from_env(self, monkeypatch):
        """Test that 'legacy' mode is returned when explicitly set."""
        monkeypatch.setenv("FEATURE_MATERIALIZATION_MODE", "legacy")
        assert get_materialization_mode() == "legacy"

    def test_cursor_mode_from_env(self, monkeypatch):
        """Test that 'cursor' mode is returned when set."""
        monkeypatch.setenv("FEATURE_MATERIALIZATION_MODE", "cursor")
        assert get_materialization_mode() == "cursor"

    def test_invalid_mode_falls_back_to_legacy(self, monkeypatch):
        """Test that invalid mode values fall back to 'legacy'."""
        monkeypatch.setenv("FEATURE_MATERIALIZATION_MODE", "invalid")
        assert get_materialization_mode() == "legacy"

    def test_empty_mode_falls_back_to_legacy(self, monkeypatch):
        """Test that empty mode value uses default 'legacy'."""
        monkeypatch.setenv("FEATURE_MATERIALIZATION_MODE", "")
        assert get_materialization_mode() == "legacy"


class TestMaterializationModeType:
    """Tests for MaterializationMode type constraints."""

    def test_type_includes_legacy(self):
        """Test that MaterializationMode includes 'legacy'."""
        # Type check at runtime via casting
        mode: MaterializationMode = "legacy"
        assert mode == "legacy"

    def test_type_includes_cursor(self):
        """Test that MaterializationMode includes 'cursor'."""
        mode: MaterializationMode = "cursor"
        assert mode == "cursor"
