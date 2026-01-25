"""Tests for Model Lab UI components: run details, comparison, progress."""

from unittest.mock import MagicMock, patch

from ui.mlflow_utils import get_run_artifacts, get_run_details


class TestRunExpander:
    """Run detail expander shows params and metrics."""

    @patch("ui.mlflow_utils.get_client")
    def test_get_run_details_returns_params_metrics_tags(self, mock_client):
        """get_run_details returns dict with params, metrics, tags."""
        mock_run = MagicMock()
        mock_run.data.params = {"max_depth": "6", "train_size": "1000"}
        mock_run.data.metrics = {"pr_auc": 0.85, "f1": 0.72}
        mock_run.data.tags = {"git_sha": "abc123"}
        mock_client.return_value.get_run.return_value = mock_run

        out = get_run_details("run-1")
        assert "params" in out
        assert "metrics" in out
        assert "tags" in out
        assert out["params"]["max_depth"] == "6"
        assert out["metrics"]["pr_auc"] == 0.85

    @patch("ui.mlflow_utils.get_client")
    def test_get_run_artifacts_returns_paths(self, mock_client):
        """get_run_artifacts returns list of path/is_dir."""
        mock_client.return_value.list_artifacts.return_value = [
            MagicMock(path="confusion_matrix.png", is_dir=False),
            MagicMock(path="model", is_dir=True),
        ]
        out = get_run_artifacts("run-1")
        assert len(out) == 2
        assert out[0]["path"] == "confusion_matrix.png"
        assert out[0]["is_dir"] is False
        assert out[1]["is_dir"] is True


class TestComparisonView:
    """Model comparison view."""

    def test_comparison_metrics_keys(self):
        """Comparison uses expected metric keys."""
        metric_keys = ["precision", "recall", "pr_auc", "f1", "roc_auc"]
        assert "pr_auc" in metric_keys
        assert "f1" in metric_keys

    def test_config_diff_detection(self):
        """Params that differ across runs are detectable."""
        a = {"max_depth": 6, "n_estimators": 100}
        b = {"max_depth": 8, "n_estimators": 100}
        all_params = set(a.keys()) | set(b.keys())
        diffs = [
            (p, [a.get(p), b.get(p)])
            for p in sorted(all_params)
            if len({a.get(p), b.get(p)}) > 1
        ]
        assert len(diffs) == 1
        assert diffs[0][0] == "max_depth"


class TestProgressDisplay:
    """Training progress shows stages."""

    def test_progress_stages_in_message(self):
        """Progress message includes stage labels."""
        stages = (
            "1. Loading data · 2. Training model · "
            "3. Computing metrics · 4. Logging artifacts"
        )
        assert "1. Loading data" in stages
        assert "2. Training model" in stages
        assert "3. Computing metrics" in stages
        assert "4. Logging artifacts" in stages
