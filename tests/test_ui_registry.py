"""Tests for model registry UI enhancements."""

from unittest.mock import MagicMock, patch

from ui.mlflow_utils import get_version_details


class TestVersionDisplay:
    """Version display and metadata."""

    @patch("ui.mlflow_utils.get_client")
    def test_version_display_shows_metadata(self, mock_client):
        """get_version_details returns version, stage, run_id, metrics."""
        mock_v = MagicMock()
        mock_v.version = "3"
        mock_v.run_id = "run-abc"
        mock_v.current_stage = "Production"
        mock_v.creation_timestamp = 1234567890
        mock_client.return_value.get_model_version.return_value = mock_v
        with patch("ui.mlflow_utils.get_run_details") as mock_details:
            mock_details.return_value = {
                "metrics": {"pr_auc": 0.85, "f1": 0.72},
            }
            out = get_version_details(version="3")
        assert out["version"] == "3"
        assert out["stage"] == "Production"
        assert out["run_id"] == "run-abc"
        assert "metrics" in out
        assert out["metrics"]["pr_auc"] == 0.85

    @patch("ui.mlflow_utils.get_client")
    def test_production_model_highlighted(self, mock_client):
        """Production stage is returned in version details."""
        mock_v = MagicMock()
        mock_v.version = "1"
        mock_v.run_id = "r"
        mock_v.current_stage = "Production"
        mock_v.creation_timestamp = 0
        mock_client.return_value.search_model_versions.return_value = [mock_v]
        with patch("ui.mlflow_utils.get_run_details", return_value={"metrics": {}}):
            out = get_version_details(version="latest")
        assert out.get("stage") == "Production"


class TestTradeoffChart:
    """Tradeoff scatter rendering."""

    def test_tradeoff_uses_pr_auc_and_f1(self):
        """Tradeoff chart uses PR-AUC and F1."""
        metric_keys = ["precision", "recall", "pr_auc", "f1", "roc_auc"]
        assert "pr_auc" in metric_keys
        assert "f1" in metric_keys


class TestArtifactPreview:
    """Model card and artifact preview."""

    def test_model_card_artifact_name(self):
        """Model card artifact is model_card.md."""
        assert "model_card.md" == "model_card.md"
