"""Tests for the feature store and materialization pipeline."""


class TestFeatureMaterializer:
    """Tests for feature materialization via Analytics service."""

    def test_materialize_function_imports(self):
        """Test that materialize_features function can be imported."""
        from training_server.materialize_features import materialize_features

        assert materialize_features is not None

    def test_materialization_mode_imports(self):
        """Test that materialization mode function can be imported."""
        from training_server.materialize_features import get_materialization_mode

        assert get_materialization_mode is not None

    def test_materialization_mode_default_is_legacy(self, monkeypatch):
        """Test that default materialization mode is legacy."""
        monkeypatch.delenv("FEATURE_MATERIALIZATION_MODE", raising=False)
        from training_server.materialize_features import get_materialization_mode

        assert get_materialization_mode() == "legacy"

    def test_materialize_features_via_analytics(self, monkeypatch):
        """Test that materialize_features calls analytics service."""
        from unittest.mock import MagicMock

        from training_server import crud_client
        from training_server.materialize_features import materialize_features

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.success = True
        mock_response.total_processed = 100
        mock_client.materialize_features.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        result = materialize_features(batch_size=500)

        assert result["success"] is True
        assert result["total_processed"] == 100
        mock_client.materialize_features.assert_called_once_with(batch_size=500)

    def test_materialize_features_handles_errors(self, monkeypatch):
        """Test that materialize_features handles analytics errors gracefully."""
        from unittest.mock import MagicMock

        from training_server import crud_client
        from training_server.materialize_features import materialize_features

        mock_client = MagicMock()
        mock_client.materialize_features.side_effect = Exception("Analytics error")
        monkeypatch.setattr(crud_client, "_client", mock_client)

        result = materialize_features()

        assert result["success"] is False
        assert "error" in result
        assert "Analytics error" in result["error"]
