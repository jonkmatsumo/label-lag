"""MLflow-enabled training pipeline for fraud detection model."""

import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import xgboost as xgb_pkg
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
from xgboost import XGBClassifier

from api.schemas import SplitConfig, SplitStrategy
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


def _save_confusion_matrix_plot(y_test, y_pred, path: str | Path) -> None:
    """Save confusion matrix as PNG heatmap."""
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_yticklabels(["Legit (0)", "Fraud (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.colorbar(im, ax=ax, label="Count")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance(
    clf, feature_names: list[str], path_base: str | Path
) -> tuple[str, str]:
    """Save feature importance as JSON and bar chart PNG. Returns paths."""
    path_base = Path(path_base)
    importances = clf.feature_importances_
    data = dict(zip(feature_names, [float(x) for x in importances]))
    json_path = path_base.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    names = list(data.keys())
    vals = list(data.values())
    ax.barh(names, vals)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    png_path = path_base.with_name(path_base.stem + "_plot.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(json_path), str(png_path)


def _generate_model_card(params: dict, metrics: dict, path: str | Path) -> None:
    """Write model card markdown with training summary, metrics, and config."""
    tr = params.get("train_fraud_rate")
    te = params.get("test_fraud_rate")
    tr_s = f"{tr:.4f}" if isinstance(tr, (int, float)) else "N/A"
    te_s = f"{te:.4f}" if isinstance(te, (int, float)) else "N/A"
    lines = [
        "# Model Card",
        "",
        "## Training Summary",
        f"- **Train size:** {params.get('train_size', 'N/A')}",
        f"- **Test size:** {params.get('test_size', 'N/A')}",
        f"- **Train fraud rate:** {tr_s}",
        f"- **Test fraud rate:** {te_s}",
        "",
        "## Metrics",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "## Config",
            f"- **max_depth:** {params.get('max_depth', 'N/A')}",
            f"- **n_estimators:** {params.get('n_estimators', 'N/A')}",
            f"- **learning_rate:** {params.get('learning_rate', 'N/A')}",
            f"- **training_window_days:** {params.get('training_window_days', 'N/A')}",
        ]
    )
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _compute_metrics(y_true, y_pred, y_proba):
    """Compute precision, recall, pr_auc, f1, roc_auc, log_loss, brier, tp/fp/tn/fn."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    n_classes = len(np.unique(y_true))
    roc_auc_val = roc_auc_score(y_true, y_proba) if n_classes > 1 else 0.0
    log_loss_val = log_loss(y_true, y_proba) if n_classes > 1 else 0.0
    brier = brier_score_loss(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tp = int(((yt == 1) & (yp == 1)).sum())
    return {
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


def train_model(
    scale_pos_weight: float | None = None,
    max_depth: int = 6,
    training_window_days: int = 30,
    database_url: str | None = None,
    feature_columns: list[str] | None = None,
    split_config: SplitConfig | None = None,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    min_child_weight: int = 1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    gamma: float = 0.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    early_stopping_rounds: int | None = None,
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
        split_config: Optional split/CV config. When strategy is KFOLD_TEMPORAL,
            per-fold metrics are logged and aggregated.
        n_estimators: Number of boosting rounds. Default 100.
        learning_rate: Step size shrinkage. Default 0.1.
        min_child_weight: Minimum sum of instance weight in a child. Default 1.
        subsample: Row subsample ratio. Default 1.0.
        colsample_bytree: Column subsample ratio. Default 1.0.
        gamma: Min loss reduction for split. Default 0.0.
        reg_alpha: L1 regularization. Default 0.0.
        reg_lambda: L2 regularization. Default 1.0.
        random_state: Random seed. Default 42.
        early_stopping_rounds: Optional early stopping rounds. Default None.

    Returns:
        The MLflow run ID.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    training_cutoff_date = datetime.now(UTC) - timedelta(days=training_window_days)

    loader = DataLoader(database_url=database_url)
    split = loader.load_train_test_split(
        training_cutoff_date,
        feature_columns=feature_columns,
        split_config=split_config,
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

        params_log: dict = {
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
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
        }
        if early_stopping_rounds is not None:
            params_log["early_stopping_rounds"] = early_stopping_rounds
        mlflow.log_params(params_log)

        clf_kw: dict = {
            "scale_pos_weight": scale_pos_weight,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
        if early_stopping_rounds is not None:
            clf_kw["early_stopping_rounds"] = early_stopping_rounds
        clf = XGBClassifier(**clf_kw)

        # Optional CV loop: log per-fold metrics and aggregates
        do_cv = (
            split_config is not None
            and split_config.strategy == SplitStrategy.KFOLD_TEMPORAL
        )
        if do_cv and split.train_size >= split_config.n_folds:
            k = split_config.n_folds
            n = split.train_size
            fold_metrics: list[dict] = []
            x_tr = split.X_train
            y_tr = split.y_train
            fold_size = n // k
            for fold_i in range(k):
                val_start = fold_i * fold_size
                val_end = n if fold_i == k - 1 else (fold_i + 1) * fold_size
                val_idx = np.arange(val_start, val_end)
                train_idx = np.concatenate(
                    [np.arange(0, val_start), np.arange(val_end, n)]
                )
                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue
                x_fold_train = x_tr.iloc[train_idx]
                y_fold_train = y_tr.iloc[train_idx]
                x_fold_val = x_tr.iloc[val_idx]
                y_fold_val = y_tr.iloc[val_idx]
                sw = scale_pos_weight
                if (y_fold_train == 1).sum() == 0:
                    continue
                if sw is None:
                    sw = float((y_fold_train == 0).sum() / (y_fold_train == 1).sum())
                fold_clf = XGBClassifier(
                    scale_pos_weight=sw,
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    min_child_weight=min_child_weight,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    gamma=gamma,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=random_state,
                    use_label_encoder=False,
                    eval_metric="logloss",
                )
                fold_clf.fit(x_fold_train, y_fold_train)
                y_vp = fold_clf.predict(x_fold_val)
                y_vprob = fold_clf.predict_proba(x_fold_val)[:, 1]
                fm = _compute_metrics(y_fold_val, y_vp, y_vprob)
                fold_metrics.append(fm)
                for key, val in fm.items():
                    mlflow.log_metric(f"cv_{key}_fold_{fold_i}", val, step=fold_i)
            if fold_metrics:
                agg = {}
                for key in fold_metrics[0]:
                    vals = [m[key] for m in fold_metrics]
                    agg[f"{key}_mean"] = float(np.mean(vals))
                    agg[f"{key}_std"] = float(np.std(vals))
                mlflow.log_metrics(agg)

        fit_kw: dict = {}
        x_fit = split.X_train
        y_fit = split.y_train
        if early_stopping_rounds is not None and split.train_size >= 20:
            v_frac = 0.2
            if split_config is not None:
                v_frac = split_config.validation_fraction
            n = split.train_size
            val_size = max(1, int(n * v_frac))
            train_size = n - val_size
            if train_size >= 10 and val_size >= 1:
                x_fit = split.X_train.iloc[:train_size]
                y_fit = split.y_train.iloc[:train_size]
                x_val = split.X_train.iloc[train_size:]
                y_val = split.y_train.iloc[train_size:]
                fit_kw["eval_set"] = [(x_val, y_val)]
        if fit_kw:
            clf.fit(x_fit, y_fit, **fit_kw)
            if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
                mlflow.log_metric("best_iteration", int(clf.best_iteration))
            if hasattr(clf, "best_score") and clf.best_score is not None:
                try:
                    mlflow.log_metric("best_score", float(clf.best_score))
                except (TypeError, ValueError):
                    pass
        else:
            clf.fit(x_fit, y_fit)

        y_pred = clf.predict(split.X_test)
        y_pred_proba = clf.predict_proba(split.X_test)[:, 1]
        metrics_dict = _compute_metrics(split.y_test, y_pred, y_pred_proba)

        mlflow.log_metrics(metrics_dict)

        # Log the model with signature and input example
        signature = infer_signature(split.X_train, y_pred_proba)
        mlflow.sklearn.log_model(
            clf,
            "model",
            signature=signature,
            input_example=split.X_train.iloc[:1],
        )

        params_dict = {
            "train_size": split.train_size,
            "test_size": split.test_size,
            "train_fraud_rate": split.train_fraud_rate,
            "test_fraud_rate": split.test_fraud_rate,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "training_window_days": training_window_days,
        }

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

            # Confusion matrix plot
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            _save_confusion_matrix_plot(split.y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path)

            # Feature importance JSON + plot
            fi_base = os.path.join(tmpdir, "feature_importance")
            fi_json, fi_png = _save_feature_importance(
                clf, actual_feature_columns, fi_base
            )
            mlflow.log_artifact(fi_json)
            mlflow.log_artifact(fi_png)

            card_path = os.path.join(tmpdir, "model_card.md")
            _generate_model_card(params_dict, metrics_dict, card_path)
            mlflow.log_artifact(card_path)

            if split.split_manifest is not None:
                manifest_path = os.path.join(tmpdir, "split_manifest.json")
                with open(manifest_path, "w") as f:
                    json.dump(split.split_manifest, f, indent=2)
                mlflow.log_artifact(manifest_path)

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
