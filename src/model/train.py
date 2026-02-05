"""MLflow-enabled training pipeline for fraud detection model."""

import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from model.loader import DataLoader

if TYPE_CHECKING:
    from api.schemas import SplitConfig, TuningConfig
    import mlflow
    import matplotlib.pyplot as plt

# Placeholders for lazy loading and patching
mlflow: Any = None
plt: Any = None
xgb_pkg: Any = None
XGBClassifier: Any = None
run_tuning_study: Any = None


def _get_mlflow():
    global mlflow
    if mlflow is None:
        import mlflow as _mlflow
        mlflow = _mlflow
    return mlflow


def _get_plt():
    global plt
    if plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt
    return plt


def _get_xgb():
    global xgb_pkg, XGBClassifier
    if xgb_pkg is None:
        import xgboost as _xgb
        xgb_pkg = _xgb
    if XGBClassifier is None:
        from xgboost import XGBClassifier as _XGBClassifier
        XGBClassifier = _XGBClassifier
    return xgb_pkg


def _get_run_tuning_study():
    global run_tuning_study
    if run_tuning_study is None:
        from model.tuning import run_tuning_study as _run_tuning_study
        run_tuning_study = _run_tuning_study
    return run_tuning_study


# experiment name
EXPERIMENT_NAME = "ach-fraud-detection"


def _get_git_sha() -> str | None:
    """Return current git commit SHA, or None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _save_confusion_matrix_plot(y_test, y_pred, path: str | Path) -> None:
    """Save confusion matrix as PNG heatmap."""
    from sklearn.metrics import confusion_matrix
    _plt = _get_plt()
    
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = _plt.subplots(figsize=(6, 5))
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
    _plt.colorbar(im, ax=ax, label="Count")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance(
    clf, feature_names: list[str], path_base: str | Path
) -> tuple[str, str]:
    """Save feature importance as JSON and bar chart PNG. Returns paths."""
    _plt = _get_plt()
    path_base = Path(path_base)
    importances = clf.feature_importances_
    data = dict(zip(feature_names, [float(x) for x in importances]))
    json_path = path_base.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    fig, ax = _plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    names = list(data.keys())
    vals = list(data.values())
    ax.barh(names, vals)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    _plt.tight_layout()
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
    import numpy as np
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
    split_config: "SplitConfig | None" = None,
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
    tuning_config: "TuningConfig | None" = None,
) -> str:
    """Train an XGBoost model with MLflow tracking."""
    _mlflow = _get_mlflow()
    _get_xgb()
    _run_tuning = _get_run_tuning_study()
    import numpy as np
    from mlflow.models import infer_signature
    from api.schemas import SplitStrategy

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    _mlflow.set_tracking_uri(tracking_uri)
    _mlflow.set_experiment(EXPERIMENT_NAME)
    
    training_cutoff_date = datetime.now(UTC) - timedelta(days=training_window_days)

    loader = DataLoader(database_url=database_url)
    split = loader.load_train_test_split(
        training_cutoff_date,
        feature_columns=feature_columns,
        split_config=split_config,
    )

    actual_feature_columns = feature_columns if feature_columns is not None else loader.FEATURE_COLUMNS

    if split.train_size == 0:
        raise ValueError("No training data available. Generate data first.")
    if split.test_size == 0:
        raise ValueError("No test data available. Adjust training_window_days.")

    n_negative = (split.y_train == 0).sum()
    n_positive = (split.y_train == 1).sum()

    if n_positive == 0:
        cutoff = training_cutoff_date.date()
        raise ValueError(f"No fraud samples in training set (cutoff: {cutoff}).")

    if scale_pos_weight is None:
        scale_pos_weight = n_negative / n_positive

    import time
    training_start_time = time.time()

    with _mlflow.start_run() as run:
        _mlflow.set_tags({
            "git_sha": _get_git_sha() or "unknown",
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "xgboost_version": xgb_pkg.__version__,
        })

        if split.split_manifest is not None:
            _mlflow.set_tags({
                "split.strategy": split.split_manifest.get("strategy", "unknown"),
                "split.train_size": str(split.split_manifest.get("train_size", 0)),
                "split.test_size": str(split.split_manifest.get("test_size", 0)),
                "split.train_fraud_rate": str(split.split_manifest.get("train_fraud_rate", 0.0)),
                "split.test_fraud_rate": str(split.split_manifest.get("test_fraud_rate", 0.0)),
            })

        trials_df = None
        if tuning_config is not None and tuning_config.enabled and split.train_size >= 30:
            v_frac = split_config.validation_fraction if split_config else 0.2
            n = split.train_size
            val_size = max(5, int(n * v_frac))
            train_size = n - val_size
            if train_size >= 10:
                x_tr = split.X_train.iloc[:train_size]
                y_tr = split.y_train.iloc[:train_size]
                x_val = split.X_train.iloc[train_size:]
                y_val = split.y_train.iloc[train_size:]
                best, trials_df = _run_tuning(
                    x_tr, y_tr, x_val, y_val,
                    n_trials=tuning_config.n_trials,
                    metric=tuning_config.metric,
                    timeout_seconds=tuning_config.timeout_minutes * 60,
                    seed=split_config.seed if split_config else 42,
                    scale_pos_weight=scale_pos_weight,
                )
                selected_params = best
                selection_type = "auto"
                selected_trial_num = None
                if tuning_config.selected_trial_number is not None:
                    from model.tuning import get_trial_params
                    manual = get_trial_params(trials_df, tuning_config.selected_trial_number)
                    if manual: 
                        selected_params = manual
                        selection_type = "manual"
                        selected_trial_num = tuning_config.selected_trial_number
                if selected_params:
                    max_depth = selected_params.get("max_depth", max_depth)
                    n_estimators = selected_params.get("n_estimators", n_estimators)
                    learning_rate = selected_params.get("learning_rate", learning_rate)
                    min_child_weight = selected_params.get("min_child_weight", min_child_weight)
                    subsample = selected_params.get("subsample", subsample)
                    colsample_bytree = selected_params.get("colsample_bytree", colsample_bytree)
                    gamma = selected_params.get("gamma", gamma)
                    reg_alpha = selected_params.get("reg_alpha", reg_alpha)
                    reg_lambda = selected_params.get("reg_lambda", reg_lambda)
                    for k, v in selected_params.items(): _mlflow.log_param(f"tuning_best_{k}", v)
                
                _mlflow.set_tags({
                    "tuning.selected_trial": str(selected_trial_num) if selected_trial_num is not None else "best",
                    "tuning.selection_type": selection_type,
                    "tuning.n_trials": str(tuning_config.n_trials),
                })

        params_log = {
            "scale_pos_weight": scale_pos_weight, "max_depth": max_depth, "training_window_days": training_window_days,
            "train_size": split.train_size, "test_size": split.test_size, "n_estimators": n_estimators,
            "learning_rate": learning_rate, "random_state": random_state,
            "feature_columns": json.dumps(actual_feature_columns),
        }
        if early_stopping_rounds: params_log["early_stopping_rounds"] = early_stopping_rounds
        _mlflow.log_params(params_log)

        clf_kw = {
            "scale_pos_weight": scale_pos_weight, "max_depth": max_depth, "n_estimators": n_estimators,
            "learning_rate": learning_rate, "min_child_weight": min_child_weight, "subsample": subsample,
            "colsample_bytree": colsample_bytree, "gamma": gamma, "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,
            "random_state": random_state, "use_label_encoder": False, "eval_metric": "logloss",
        }
        if early_stopping_rounds: clf_kw["early_stopping_rounds"] = early_stopping_rounds
        clf = XGBClassifier(**clf_kw)

        # CV loop
        do_cv = split_config and split_config.strategy == SplitStrategy.KFOLD_TEMPORAL
        if do_cv and split.train_size >= split_config.n_folds:
            k = split_config.n_folds
            n = split.train_size
            fold_metrics = []
            x_tr = split.X_train
            y_tr = split.y_train
            fold_size = n // k
            for fold_i in range(k):
                val_start = fold_i * fold_size
                val_end = n if fold_i == k-1 else (fold_i+1)*fold_size
                val_idx = np.arange(val_start, val_end)
                train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
                if not len(train_idx) or not len(val_idx): continue
                fold_clf = XGBClassifier(**clf_kw)
                fold_clf.fit(x_tr.iloc[train_idx], y_tr.iloc[train_idx])
                y_vp = fold_clf.predict(x_tr.iloc[val_idx])
                y_vprob = fold_clf.predict_proba(x_tr.iloc[val_idx])[:, 1]
                fm = _compute_metrics(y_tr.iloc[val_idx], y_vp, y_vprob)
                fold_metrics.append(fm)
                for key, val in fm.items(): _mlflow.log_metric(f"cv_{key}_fold_{fold_i}", val, step=fold_i)
            if fold_metrics:
                agg = {}
                for key in fold_metrics[0]:
                    vals = [m[key] for m in fold_metrics]
                    agg[f"{key}_mean"] = float(np.mean(vals))
                    agg[f"{key}_std"] = float(np.std(vals))
                    agg[f"{key}_min"] = float(np.min(vals))
                    agg[f"{key}_max"] = float(np.max(vals))
                _mlflow.log_metrics(agg)
                _mlflow.set_tags({"cv.enabled": "true", "cv.n_folds": str(k)})

        x_fit, y_fit = split.X_train, split.y_train
        if early_stopping_rounds is not None and split.train_size >= 20:
            v_frac = split_config.validation_fraction if split_config else 0.2
            n = split.train_size
            val_size = max(1, int(n * v_frac))
            train_size = n - val_size
            if train_size >= 10 and val_size >= 1:
                x_fit = split.X_train.iloc[:train_size]
                y_fit = split.y_train.iloc[:train_size]
                x_val = split.X_train.iloc[train_size:]
                y_val = split.y_train.iloc[train_size:]
                clf.fit(x_fit, y_fit, eval_set=[(x_val, y_val)])
                if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
                    _mlflow.log_metric("best_iteration", int(clf.best_iteration))
            else:
                clf.fit(x_fit, y_fit)
        else:
            clf.fit(x_fit, y_fit)

        y_pred = clf.predict(split.X_test)
        y_pred_proba = clf.predict_proba(split.X_test)[:, 1]
        metrics_dict = _compute_metrics(split.y_test, y_pred, y_pred_proba)
        _mlflow.log_metrics(metrics_dict)

        signature = infer_signature(split.X_train, y_pred_proba)
        _mlflow.sklearn.log_model(clf, "model", signature=signature, input_example=split.X_train.iloc[:1])

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "reference_data.parquet")
            split.X_test.to_parquet(ref_path, index=False)
            _mlflow.log_artifact(ref_path)
            
            # Save feature columns list as artifact for inference
            feature_columns_path = os.path.join(tmpdir, "feature_columns.json")
            with open(feature_columns_path, "w") as f:
                json.dump(actual_feature_columns, f, indent=2)
            _mlflow.log_artifact(feature_columns_path)

            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            _save_confusion_matrix_plot(split.y_test, y_pred, cm_path)
            _mlflow.log_artifact(cm_path)

            fi_base = os.path.join(tmpdir, "feature_importance")
            fi_json, fi_png = _save_feature_importance(clf, actual_feature_columns, fi_base)
            _mlflow.log_artifact(fi_json)
            _mlflow.log_artifact(fi_png)

        _mlflow.log_metric("training_time_seconds", time.time() - training_start_time)
        model_uri = f"runs:/{run.info.run_id}/model"
        _mlflow.register_model(model_uri, EXPERIMENT_NAME)
        return run.info.run_id


def get_latest_model_version(model_name: str = EXPERIMENT_NAME) -> int | None:
    """Get the latest version number of a registered model."""
    _mlflow = _get_mlflow()
    client = _mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions: return max(int(v.version) for v in versions)
    except Exception: pass
    return None


def load_production_model(model_name: str = EXPERIMENT_NAME):
    """Load the latest version of the production model."""
    _mlflow = _get_mlflow()
    import mlflow.sklearn
    version = get_latest_model_version(model_name)
    if version is None: raise ValueError(f"No model versions found for '{model_name}'")
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(model_uri)