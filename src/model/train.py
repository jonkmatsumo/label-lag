"""MLflow-enabled training pipeline for fraud detection model."""

import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb_pkg
from xgboost import XGBClassifier

from model.loader import DataLoader

# MLflow configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Experiment name
EXPERIMENT_NAME = "ach-fraud-detection"


def _get_git_sha() -> str | None:
    """Return current git commit SHA, or None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def train_model(
    scale_pos_weight: float | None = None,
    max_depth: int = 6,
    training_window_days: int = 30,
    database_url: str | None = None,
    feature_columns: list[str] | None = None,
) -> str:
    """Train an XGBoost model with MLflow tracking.

    Args:
        scale_pos_weight: Weight for positive class. If None, computed automatically
            from class imbalance ratio.
        max_depth: Maximum tree depth. Default 6.
        training_window_days: Number of days before today for training cutoff.
            Default 30.
        database_url: Optional database URL override.
        feature_columns: Optional list of feature columns to use. If None, uses
            default FEATURE_COLUMNS from DataLoader.

    Returns:
        The MLflow run ID.
    """
    # Set up MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Calculate training cutoff date
    training_cutoff_date = datetime.now(UTC) - timedelta(days=training_window_days)

    # Load data
    loader = DataLoader(database_url=database_url)
    split = loader.load_train_test_split(
        training_cutoff_date, feature_columns=feature_columns
    )

    # Determine actual feature columns used (for logging)
    actual_feature_columns = (
        feature_columns if feature_columns is not None else loader.FEATURE_COLUMNS
    )

    # Handle empty dataset
    if split.train_size == 0:
        raise ValueError("No training data available. Generate data first.")

    if split.test_size == 0:
        raise ValueError("No test data available. Adjust training_window_days.")

    # Check for positive samples in training set
    n_negative = (split.y_train == 0).sum()
    n_positive = (split.y_train == 1).sum()

    if n_positive == 0:
        cutoff = training_cutoff_date.date()
        raise ValueError(
            f"No fraud samples in training set (cutoff: {cutoff}). "
            f"Try a smaller training_window_days (current: {training_window_days}), "
            "or regenerate data with: docker compose run --rm generator "
            "uv run python src/main.py seed --drop-tables"
        )

    # Calculate scale_pos_weight if not provided
    if scale_pos_weight is None:
        scale_pos_weight = n_negative / n_positive

    with mlflow.start_run() as run:
        # Log run metadata as tags
        mlflow.set_tags(
            {
                "git_sha": _get_git_sha() or "unknown",
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "xgboost_version": xgb_pkg.__version__,
            }
        )

        # Log parameters (including previously hardcoded hyperparams)
        n_estimators = 100
        learning_rate = 0.1
        random_state = 42
        mlflow.log_params(
            {
                "scale_pos_weight": scale_pos_weight,
                "max_depth": max_depth,
                "training_window_days": training_window_days,
                "training_cutoff_date": training_cutoff_date.isoformat(),
                "train_size": split.train_size,
                "test_size": split.test_size,
                "train_fraud_rate": split.train_fraud_rate,
                "test_fraud_rate": split.test_fraud_rate,
                "feature_columns": json.dumps(actual_feature_columns),
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "random_state": random_state,
            }
        )

        # Train XGBoost model
        clf = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )

        clf.fit(split.X_train, split.y_train)

        # Generate predictions
        y_pred = clf.predict(split.X_test)
        y_pred_proba = clf.predict_proba(split.X_test)[:, 1]

        # Calculate metrics
        precision = precision_score(split.y_test, y_pred, zero_division=0)
        recall = recall_score(split.y_test, y_pred, zero_division=0)
        pr_auc = average_precision_score(split.y_test, y_pred_proba)
        f1 = f1_score(split.y_test, y_pred, zero_division=0)
        roc_auc_val = (
            roc_auc_score(split.y_test, y_pred_proba)
            if len(split.y_test.unique()) > 1
            else 0.0
        )
        log_loss_val = (
            log_loss(split.y_test, y_pred_proba)
            if len(split.y_test.unique()) > 1
            else 0.0
        )
        brier = brier_score_loss(split.y_test, y_pred_proba)
        cm = confusion_matrix(split.y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            yt = split.y_test.values
            yp = y_pred
            tn = int(((yt == 0) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            tp = int(((yt == 1) & (yp == 1)).sum())

        # Log metrics
        mlflow.log_metrics(
            {
                "precision": precision,
                "recall": recall,
                "pr_auc": pr_auc,
                "f1": f1,
                "roc_auc": roc_auc_val,
                "log_loss": log_loss_val,
                "brier_score": brier,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

        # Log the model with signature and input example
        signature = infer_signature(split.X_train, y_pred_proba)
        mlflow.sklearn.log_model(
            clf,
            "model",
            signature=signature,
            input_example=split.X_train.iloc[:1],
        )

        # Save and log reference data (X_test) for drift detection
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path = os.path.join(tmpdir, "reference_data.parquet")
            split.X_test.to_parquet(reference_path, index=False)
            mlflow.log_artifact(reference_path)

            # Save feature columns list as artifact for inference
            feature_columns_path = os.path.join(tmpdir, "feature_columns.json")
            with open(feature_columns_path, "w") as f:
                json.dump(actual_feature_columns, f, indent=2)
            mlflow.log_artifact(feature_columns_path)

        # Register the model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, EXPERIMENT_NAME)

        return run.info.run_id


def get_latest_model_version(model_name: str = EXPERIMENT_NAME) -> int | None:
    """Get the latest version number of a registered model.

    Args:
        model_name: Name of the registered model.

    Returns:
        Latest version number, or None if model doesn't exist.
    """
    client = mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            return max(int(v.version) for v in versions)
    except mlflow.exceptions.MlflowException:
        pass
    return None


def load_production_model(model_name: str = EXPERIMENT_NAME):
    """Load the latest version of the production model.

    Args:
        model_name: Name of the registered model.

    Returns:
        Loaded model object.

    Raises:
        ValueError: If no model versions exist.
    """
    version = get_latest_model_version(model_name)
    if version is None:
        raise ValueError(f"No model versions found for '{model_name}'")

    model_uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(model_uri)


if __name__ == "__main__":
    import sys

    # Allow overriding training window from command line
    window_days = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    print(f"Training model with {window_days} day window...")
    run_id = train_model(training_window_days=window_days)
    print(f"Training complete. Run ID: {run_id}")
