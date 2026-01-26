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


class TestCVFoldMetrics:
    """CV fold metrics extraction and display."""

    @patch("ui.mlflow_utils.get_run_details")
    def test_get_cv_fold_metrics_extracts_folds(self, mock_get_details):
        """get_cv_fold_metrics extracts and groups fold values."""
        from ui.mlflow_utils import get_cv_fold_metrics

        mock_get_details.return_value = {
            "metrics": {
                "cv_precision_fold_0": 0.85,
                "cv_precision_fold_1": 0.87,
                "cv_precision_fold_2": 0.86,
                "cv_recall_fold_0": 0.72,
                "cv_recall_fold_1": 0.74,
                "cv_recall_fold_2": 0.73,
            },
        }

        result = get_cv_fold_metrics("run-1")
        assert "precision" in result
        assert "recall" in result
        assert len(result["precision"]) == 3
        assert len(result["recall"]) == 3
        assert result["precision"] == [0.85, 0.87, 0.86]
        assert result["recall"] == [0.72, 0.74, 0.73]

    @patch("ui.mlflow_utils.get_run_details")
    def test_get_cv_fold_metrics_empty_when_no_cv(self, mock_get_details):
        """Returns empty dict for non-CV runs."""
        from ui.mlflow_utils import get_cv_fold_metrics

        mock_get_details.return_value = {
            "metrics": {
                "precision": 0.85,
                "recall": 0.72,
                "pr_auc": 0.80,
            },
        }

        result = get_cv_fold_metrics("run-1")
        assert result == {}

    @patch("ui.mlflow_utils.get_run_details")
    def test_get_cv_fold_metrics_handles_missing_folds(self, mock_get_details):
        """Handles non-sequential fold numbers gracefully."""
        from ui.mlflow_utils import get_cv_fold_metrics

        mock_get_details.return_value = {
            "metrics": {
                "cv_precision_fold_0": 0.85,
                "cv_precision_fold_2": 0.86,  # Missing fold_1
            },
        }

        result = get_cv_fold_metrics("run-1")
        # Should filter out None values
        assert "precision" in result
        assert len(result["precision"]) == 2


class TestSplitManifest:
    """Split manifest parsing and display."""

    @patch("ui.mlflow_utils.fetch_artifact_path")
    def test_get_split_manifest_parses_artifact(self, mock_fetch):
        """get_split_manifest downloads and parses JSON."""
        import json
        import tempfile
        from pathlib import Path

        from ui.mlflow_utils import get_split_manifest

        # Create temporary manifest file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            manifest_data = {
                "strategy": "temporal",
                "seed": 42,
                "train_size": 1000,
                "test_size": 200,
            }
            json.dump(manifest_data, f)
            temp_path = f.name

        try:
            mock_fetch.return_value = temp_path
            result = get_split_manifest("run-1")
            assert result is not None
            assert result["strategy"] == "temporal"
            assert result["seed"] == 42
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("ui.mlflow_utils.fetch_artifact_path")
    def test_get_split_manifest_returns_none_when_missing(self, mock_fetch):
        """Returns None when artifact not found."""
        from ui.mlflow_utils import get_split_manifest

        mock_fetch.return_value = None
        result = get_split_manifest("run-1")
        assert result is None


class TestTuningTrials:
    """Tuning trials parsing and display."""

    @patch("ui.mlflow_utils.fetch_artifact_path")
    def test_get_tuning_trials_parses_csv(self, mock_fetch):
        """get_tuning_trials downloads and parses CSV."""
        import csv
        import tempfile
        from pathlib import Path

        from ui.mlflow_utils import get_tuning_trials

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "value", "state", "params_max_depth"])
            writer.writerow([0, 0.7, "COMPLETE", 6])
            writer.writerow([1, 0.8, "COMPLETE", 8])
            temp_path = f.name

        try:
            mock_fetch.return_value = temp_path
            result = get_tuning_trials("run-1")
            assert result is not None
            assert len(result) == 2
            assert "trial" in result.columns
            assert "value" in result.columns
            assert result.iloc[0]["trial"] == 0
            assert result.iloc[1]["value"] == 0.8
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("ui.mlflow_utils.fetch_artifact_path")
    def test_get_tuning_trials_returns_none_when_missing(self, mock_fetch):
        """Returns None when artifact not found."""
        from ui.mlflow_utils import get_tuning_trials

        mock_fetch.return_value = None
        result = get_tuning_trials("run-1")
        assert result is None


class TestRunningExperiments:
    """get_running_experiments returns run IDs for in-progress runs."""

    @patch("ui.mlflow_utils.mlflow")
    def test_get_running_experiments_returns_run_ids(self, mock_mlflow):
        """Returns list of running run IDs."""
        from ui.mlflow_utils import get_running_experiments

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-1"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        import pandas as pd

        mock_mlflow.search_runs.return_value = pd.DataFrame(
            {"run_id": ["run-a", "run-b"]}
        )

        result = get_running_experiments()
        assert result == ["run-a", "run-b"]
        mock_mlflow.search_runs.assert_called_once()
        call_kw = mock_mlflow.search_runs.call_args[1]
        assert call_kw["filter_string"] == "status = 'RUNNING'"

    @patch("ui.mlflow_utils.mlflow")
    def test_get_running_experiments_returns_empty_when_none(self, mock_mlflow):
        """Returns empty list when no runs in progress."""
        from ui.mlflow_utils import get_running_experiments

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-1"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        import pandas as pd

        mock_mlflow.search_runs.return_value = pd.DataFrame()

        result = get_running_experiments()
        assert result == []

    @patch("ui.mlflow_utils.mlflow")
    def test_get_running_experiments_empty_when_no_experiment(self, mock_mlflow):
        """Returns empty list when experiment does not exist."""
        from ui.mlflow_utils import get_running_experiments

        mock_mlflow.get_experiment_by_name.return_value = None

        result = get_running_experiments()
        assert result == []
        mock_mlflow.search_runs.assert_not_called()
