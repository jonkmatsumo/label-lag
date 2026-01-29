"""MLflow utilities for the dashboard.

Provides helper functions for interacting with MLflow tracking server
and model registry. Keeps UI code clean by encapsulating MLflow client logic.
"""

import os
from typing import Any

import mlflow
import pandas as pd
import requests
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

# MLflow configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# API configuration (for model reload)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8100")

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Experiment name (must match train.py)
EXPERIMENT_NAME = "ach-fraud-detection"


def get_client() -> MlflowClient:
    """Get an MLflow client instance.

    Returns:
        MlflowClient configured with tracking URI.
    """
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_experiment_runs(
    experiment_name: str = EXPERIMENT_NAME,
    max_results: int = 100,
) -> pd.DataFrame:
    """Fetch experiment runs sorted by PR-AUC.

    Args:
        experiment_name: Name of the MLflow experiment.
        max_results: Maximum number of runs to return.

    Returns:
        DataFrame with run information, sorted by metrics.pr_auc descending.
        Empty DataFrame if experiment doesn't exist or has no runs.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
            order_by=["metrics.pr_auc DESC"],
        )

        if runs.empty:
            return pd.DataFrame()

        # Select and rename relevant columns
        columns_map = {
            "run_id": "Run ID",
            "start_time": "Started",
            "params.max_depth": "Max Depth",
            "params.training_window_days": "Window (days)",
            "params.train_size": "Train Size",
            "metrics.precision": "Precision",
            "metrics.recall": "Recall",
            "metrics.pr_auc": "PR-AUC",
            "metrics.f1": "F1",
            "metrics.roc_auc": "ROC-AUC",
        }

        available_cols = [c for c in columns_map.keys() if c in runs.columns]
        result = runs[available_cols].copy()
        result = result.rename(columns={k: v for k, v in columns_map.items()})

        return result

    except MlflowException:
        return pd.DataFrame()


def get_model_versions(
    model_name: str = EXPERIMENT_NAME,
) -> list[dict[str, Any]]:
    """Get all versions of a registered model.

    Args:
        model_name: Name of the registered model.

    Returns:
        List of version dictionaries with version, stage, and run_id.
    """
    client = get_client()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return [
            {
                "version": v.version,
                "stage": v.current_stage,
                "run_id": v.run_id,
                "created": v.creation_timestamp,
            }
            for v in versions
        ]
    except MlflowException:
        return []


def get_version_details(
    model_name: str = EXPERIMENT_NAME,
    version: str | int = "latest",
) -> dict[str, Any]:
    """Get version metadata, run_id, and key metrics from source run.

    Returns:
        Dict with version, stage, run_id, created, metrics (pr_auc, f1, etc.).
        Empty dict on error.
    """
    try:
        client = get_client()
        if version == "latest":
            vers = client.search_model_versions(f"name='{model_name}'")
            vers = sorted(vers, key=lambda v: int(v.version), reverse=True)
            if not vers:
                return {}
            v = vers[0]
        else:
            v = client.get_model_version(model_name, str(version))
        run_id = v.run_id
        details = get_run_details(run_id)
        return {
            "version": v.version,
            "stage": v.current_stage or "None",
            "run_id": run_id,
            "created": v.creation_timestamp,
            "metrics": details.get("metrics", {}),
        }
    except MlflowException:
        return {}


def get_production_model_version(model_name: str = EXPERIMENT_NAME) -> str | None:
    """Get the version number of the current production model.

    Args:
        model_name: Name of the registered model.

    Returns:
        Version string if a production model exists, None otherwise.
    """
    versions = get_model_versions(model_name)
    for v in versions:
        if v["stage"] == "Production":
            return v["version"]
    return None


def get_staging_model_version(
    model_name: str = EXPERIMENT_NAME,
) -> str | None:
    """Get the version number of the current staging model.

    Args:
        model_name: Name of the registered model.

    Returns:
        Version string if a staging model exists, None otherwise.
    """
    versions = get_model_versions(model_name)
    for v in versions:
        if v["stage"] == "Staging":
            return v["version"]
    return None


def check_promotion_thresholds(
    run_id: str, thresholds: dict[str, float]
) -> tuple[bool, list[str]]:
    """Check if run metrics meet promotion thresholds.

    Args:
        run_id: MLflow run ID.
        thresholds: Dictionary mapping metric names to minimum values.

    Returns:
        (passed, failures) tuple where failures is list of failed metric names.
    """
    if not thresholds:
        return True, []

    details = get_run_details(run_id)
    metrics = details.get("metrics", {})
    failures = []

    for metric_name, min_value in thresholds.items():
        metric_value = metrics.get(metric_name)
        if metric_value is None:
            failures.append(f"{metric_name} (not found)")
        elif metric_value < min_value:
            failures.append(f"{metric_name} ({metric_value:.4f} < {min_value:.4f})")

    return len(failures) == 0, failures


def promote_to_staging(
    run_id: str,
    model_name: str = EXPERIMENT_NAME,
) -> dict[str, Any]:
    """Promote a model run to staging stage.

    Args:
        run_id: The MLflow run ID to promote.
        model_name: Name of the registered model.

    Returns:
        Dictionary with success status and message.
    """
    client = get_client()

    try:
        # Find the version associated with this run
        versions = client.search_model_versions(f"name='{model_name}'")
        target_version = None

        for v in versions:
            if v.run_id == run_id:
                target_version = v.version
                break

        if target_version is None:
            return {
                "success": False,
                "message": f"No model version found for run {run_id}",
            }

        # Archive current staging model if exists
        for v in versions:
            if v.current_stage == "Staging":
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived",
                )

        # Promote the target version to staging
        client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Staging",
        )

        return {
            "success": True,
            "message": f"Model version {target_version} promoted to Staging.",
            "version": target_version,
        }

    except MlflowException as e:
        return {
            "success": False,
            "message": f"MLflow error: {e}",
        }


def promote_to_production(
    run_id: str,
    model_name: str = EXPERIMENT_NAME,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Promote a model run to production stage.

    Requires model to be in Staging first. Checks optional metric thresholds.

    Args:
        run_id: The MLflow run ID to promote.
        model_name: Name of the registered model.
        thresholds: Optional metric thresholds to validate.

    Returns:
        Dictionary with success status and message.
    """
    client = get_client()

    try:
        # Find the version associated with this run
        versions = client.search_model_versions(f"name='{model_name}'")
        target_version = None

        for v in versions:
            if v.run_id == run_id:
                target_version = v.version
                # Check if already in Staging
                if v.current_stage != "Staging":
                    return {
                        "success": False,
                        "message": (
                            f"Model version {target_version} must be in Staging "
                            "before promotion to Production."
                        ),
                    }
                break

        if target_version is None:
            return {
                "success": False,
                "message": f"No model version found for run {run_id}",
            }

        # Check thresholds if provided
        if thresholds:
            passed, failures = check_promotion_thresholds(run_id, thresholds)
            if not passed:
                return {
                    "success": False,
                    "message": f"Threshold check failed: {', '.join(failures)}",
                    "failures": failures,
                }

        # Archive current production model if exists
        for v in versions:
            if v.current_stage == "Production":
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived",
                )

        # Promote the target version to production
        client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Production",
        )

        # Note: Model is now in Production stage but not yet deployed
        # Use POST /models/deploy to actually deploy it

        return {
            "success": True,
            "message": (
                f"Model version {target_version} approved for Production. "
                "Use the Deploy button to deploy it to live traffic."
            ),
            "version": target_version,
        }

    except MlflowException as e:
        return {
            "success": False,
            "message": f"MLflow error: {e}",
        }


def _trigger_api_model_reload() -> str:
    """Trigger the API to reload the production model.

    Returns:
        Status message about the reload.
    """
    try:
        response = requests.post(f"{API_BASE_URL}/reload-model", timeout=30)
        result = response.json()

        if result.get("success"):
            version = result.get("version", "unknown")
            return f"API reloaded model {version}."
        else:
            return "API model reload failed."
    except Exception as e:
        return f"Could not trigger API reload: {e}"


def deploy_model(actor: str, reason: str | None = None) -> dict[str, Any]:
    """Deploy the production model to live traffic.

    Args:
        actor: Who is deploying the model.
        reason: Optional reason for deployment.

    Returns:
        Dictionary with success status and message.
    """
    url = f"{API_BASE_URL}/models/deploy"

    payload = {"actor": actor}
    if reason:
        payload["reason"] = reason

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return {
            "success": result.get("success", False),
            "message": (
                f"Model {result.get('model_version', 'unknown')} "
                f"deployed successfully. "
                f"Previous version: {result.get('previous_version', 'none')}"
            ),
            "model_version": result.get("model_version"),
            "previous_version": result.get("previous_version"),
        }
    except requests.RequestException as e:
        return {
            "success": False,
            "message": f"Deployment failed: {e}",
        }


def get_run_details(run_id: str) -> dict[str, Any]:
    """Fetch all params, metrics, and tags for a run.

    Returns:
        Dict with keys params, metrics, tags. Empty dict on error.
    """
    try:
        client = get_client()
        r = client.get_run(run_id)
        return {
            "params": dict(r.data.params),
            "metrics": dict(r.data.metrics),
            "tags": dict(r.data.tags),
        }
    except MlflowException:
        return {"params": {}, "metrics": {}, "tags": {}}


def get_cv_fold_metrics(run_id: str) -> dict[str, list[float]]:
    """Extract per-fold CV metrics from a run.

    Filters metrics matching pattern cv_{metric}_fold_{n} and groups by metric name.

    Args:
        run_id: MLflow run ID.

    Returns:
        Dictionary mapping metric names to lists of per-fold values.
        Example: {"precision": [0.85, 0.87, 0.86], "recall": [0.72, 0.74, 0.73]}
        Empty dict if no CV metrics found or run doesn't exist.
    """
    try:
        details = get_run_details(run_id)
        metrics = details.get("metrics", {})
        if not metrics:
            return {}

        # Extract CV fold metrics
        cv_metrics: dict[str, list[float]] = {}
        for metric_key, value in metrics.items():
            if metric_key.startswith("cv_") and "_fold_" in metric_key:
                # Parse: cv_{metric}_fold_{n}
                parts = metric_key.split("_fold_")
                if len(parts) == 2:
                    metric_name = parts[0][3:]  # Remove "cv_" prefix
                    try:
                        fold_num = int(parts[1])
                        if metric_name not in cv_metrics:
                            cv_metrics[metric_name] = []
                        # Ensure list is large enough
                        while len(cv_metrics[metric_name]) <= fold_num:
                            cv_metrics[metric_name].append(None)
                        cv_metrics[metric_name][fold_num] = float(value)
                    except (ValueError, TypeError):
                        continue

        # Remove None values and ensure lists are complete
        result = {}
        for metric_name, fold_values in cv_metrics.items():
            # Filter out None values and ensure all folds present
            complete_values = [v for v in fold_values if v is not None]
            if complete_values:
                result[metric_name] = complete_values

        return result
    except Exception:
        return {}


def get_run_artifacts(run_id: str) -> list[dict[str, Any]]:
    """List artifacts for a run.

    Returns:
        List of dicts with path, is_dir. Empty list on error.
    """
    try:
        client = get_client()
        items = client.list_artifacts(run_id)
        return [{"path": a.path, "is_dir": a.is_dir} for a in items]
    except MlflowException:
        return []


def fetch_artifact_path(run_id: str, artifact_path: str) -> str | None:
    """Download an artifact and return local path.

    Returns:
        Local path to the artifact, or None on error.
    """
    try:
        client = get_client()
        return client.download_artifacts(run_id, artifact_path)
    except MlflowException:
        return None


def get_split_manifest(run_id: str) -> dict[str, Any] | None:
    """Download and parse split_manifest.json artifact.

    Args:
        run_id: MLflow run ID.

    Returns:
        Parsed manifest dictionary, or None if artifact not found.
    """
    try:
        import json

        artifact_path = fetch_artifact_path(run_id, "split_manifest.json")
        if artifact_path is None:
            return None
        with open(artifact_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_tuning_trials(run_id: str) -> pd.DataFrame | None:
    """Download and parse tuning_trials.csv artifact.

    Args:
        run_id: MLflow run ID.

    Returns:
        DataFrame with trial history, or None if artifact not found.
        Columns: trial, value, state, params_* (one per hyperparameter).
    """
    try:
        artifact_path = fetch_artifact_path(run_id, "tuning_trials.csv")
        if artifact_path is None:
            return None
        df = pd.read_csv(artifact_path)
        return df
    except Exception:
        return None


def get_running_experiments(
    experiment_name: str = EXPERIMENT_NAME,
) -> list[str]:
    """Get list of run IDs for experiments currently running.

    Args:
        experiment_name: Name of the MLflow experiment.

    Returns:
        List of run IDs with status=RUNNING. Empty list if none found.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []

        # Search for running experiments
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'RUNNING'",
            max_results=100,
        )

        if runs.empty:
            return []

        return runs["run_id"].tolist()
    except Exception:
        return []


def check_mlflow_connection() -> bool:
    """Check if MLflow tracking server is accessible.

    Returns:
        True if connection successful, False otherwise.
    """
    try:
        client = get_client()
        # Try to list experiments as a connection test
        client.search_experiments(max_results=1)
        return True
    except Exception:
        return False
