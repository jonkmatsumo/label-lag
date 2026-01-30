"""Tests for inference backend configuration.

Tests the INFERENCE_BACKEND environment variable and backend selection.
"""

from api.inference_backend_config import (
    InferenceBackend,
    get_inference_backend,
    is_fallback_enabled,
    is_go_backend_enabled,
)


class TestInferenceBackendConfig:
    """Tests for inference backend configuration."""

    def test_default_backend_is_python(self, monkeypatch):
        """Test that default backend is 'python' when env var not set."""
        monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
        assert get_inference_backend() == "python"

    def test_python_backend_from_env(self, monkeypatch):
        """Test that 'python' backend is returned when explicitly set."""
        monkeypatch.setenv("INFERENCE_BACKEND", "python")
        assert get_inference_backend() == "python"

    def test_go_backend_from_env(self, monkeypatch):
        """Test that 'go' backend is returned when set."""
        monkeypatch.setenv("INFERENCE_BACKEND", "go")
        assert get_inference_backend() == "go"

    def test_go_with_fallback_from_env(self, monkeypatch):
        """Test that 'go_with_fallback' backend is returned when set."""
        monkeypatch.setenv("INFERENCE_BACKEND", "go_with_fallback")
        assert get_inference_backend() == "go_with_fallback"

    def test_invalid_backend_falls_back_to_python(self, monkeypatch):
        """Test that invalid backend values fall back to 'python'."""
        monkeypatch.setenv("INFERENCE_BACKEND", "invalid")
        assert get_inference_backend() == "python"

    def test_empty_backend_falls_back_to_python(self, monkeypatch):
        """Test that empty backend value uses default 'python'."""
        monkeypatch.setenv("INFERENCE_BACKEND", "")
        assert get_inference_backend() == "python"


class TestIsGoBackendEnabled:
    """Tests for is_go_backend_enabled helper."""

    def test_python_backend_not_go_enabled(self, monkeypatch):
        """Test that python backend returns False for is_go_backend_enabled."""
        monkeypatch.setenv("INFERENCE_BACKEND", "python")
        assert is_go_backend_enabled() is False

    def test_go_backend_is_go_enabled(self, monkeypatch):
        """Test that go backend returns True for is_go_backend_enabled."""
        monkeypatch.setenv("INFERENCE_BACKEND", "go")
        assert is_go_backend_enabled() is True

    def test_go_with_fallback_is_go_enabled(self, monkeypatch):
        """Test that go_with_fallback returns True for is_go_backend_enabled."""
        monkeypatch.setenv("INFERENCE_BACKEND", "go_with_fallback")
        assert is_go_backend_enabled() is True

    def test_default_not_go_enabled(self, monkeypatch):
        """Test that default (no env var) returns False for is_go_backend_enabled."""
        monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
        assert is_go_backend_enabled() is False


class TestIsFallbackEnabled:
    """Tests for is_fallback_enabled helper."""

    def test_python_backend_no_fallback(self, monkeypatch):
        """Test that python backend returns False for is_fallback_enabled."""
        monkeypatch.setenv("INFERENCE_BACKEND", "python")
        assert is_fallback_enabled() is False

    def test_go_backend_no_fallback(self, monkeypatch):
        """Test that go backend returns False for is_fallback_enabled."""
        monkeypatch.setenv("INFERENCE_BACKEND", "go")
        assert is_fallback_enabled() is False

    def test_go_with_fallback_has_fallback(self, monkeypatch):
        """Test that go_with_fallback returns True for is_fallback_enabled."""
        monkeypatch.setenv("INFERENCE_BACKEND", "go_with_fallback")
        assert is_fallback_enabled() is True

    def test_default_no_fallback(self, monkeypatch):
        """Test that default returns False for is_fallback_enabled."""
        monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
        assert is_fallback_enabled() is False


class TestInferenceBackendType:
    """Tests for InferenceBackend type constraints."""

    def test_type_includes_python(self):
        """Test that InferenceBackend includes 'python'."""
        backend: InferenceBackend = "python"
        assert backend == "python"

    def test_type_includes_go(self):
        """Test that InferenceBackend includes 'go'."""
        backend: InferenceBackend = "go"
        assert backend == "go"

    def test_type_includes_go_with_fallback(self):
        """Test that InferenceBackend includes 'go_with_fallback'."""
        backend: InferenceBackend = "go_with_fallback"
        assert backend == "go_with_fallback"
