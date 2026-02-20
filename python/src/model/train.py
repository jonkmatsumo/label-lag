"""MLflow-enabled training pipeline for fraud detection model."""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from model.loader import DataLoader

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import mlflow

    from training.schemas import SplitConfig, TuningConfig

# Placeholders for lazy loading and patching
mlflow: Any = None
plt: Any = None
xgb_pkg: Any = None
XGBClassifier: Any = None
run_tuning_study: Any = None
logger = logging.getLogger(__name__)


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


def _safe_run_id(run_obj, mlflow_obj) -> str:
    """Safely extract a string run_id from run object or active run."""
    try:
        rid = getattr(getattr(run_obj, "info", None), "run_id", None)
        if isinstance(rid, str) and rid:
            return rid
    except Exception:
        pass

    try:
        ar = getattr(mlflow_obj, "active_run", None)
        if callable(ar):
            active = ar()
            rid2 = getattr(getattr(active, "info", None), "run_id", None)
            if isinstance(rid2, str) and rid2:
                return rid2
    except Exception:
        pass

    # If mocked, rid may be MagicMock; don't pass that to pydantic.
    return "unknown"


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
    """Compute precision, recall, pr_auc, f1, roc_auc, log_loss, brier, tp/fp/tn/fn.

    Returns None for roc_auc and log_loss if y_true contains only one class,
    matching mathematical undefinedness for these metrics.
    """
    import numpy as np
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

    y_true = np.asarray(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    n_classes = len(np.unique(y_true))
    # AP requires at least one positive sample to be well-defined for PR curve
    pr_auc = (
        average_precision_score(y_true, y_proba)
        if n_classes > 1 and np.any(y_true == 1)
        else 0.0
    )

    f1 = f1_score(y_true, y_pred, zero_division=0)

    # These metrics are mathematically undefined for single-class folds
    roc_auc_val = roc_auc_score(y_true, y_proba) if n_classes > 1 else None
    log_loss_val = log_loss(y_true, y_proba) if n_classes > 1 else None

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


def _to_python_type(obj):
    """Convert numpy types to python types for JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _derive_calibration_split(
    x_train,
    y_train,
    validation_fraction: float,
):
    """Derive deterministic train/calibration split from train tail.

    Uses tail slicing to preserve chronology and degrades gracefully for small
    datasets by keeping enough samples/classes in fit data for model training.
    """

    def _slice(obj, start: int, end: int):
        if hasattr(obj, "iloc"):
            return obj.iloc[start:end]
        return obj[start:end]

    def _nunique(values) -> int:
        if hasattr(values, "nunique"):
            return int(values.nunique())
        import numpy as np

        return int(np.unique(values).size)

    n_samples = len(y_train)
    if n_samples == 0:
        return (
            x_train,
            y_train,
            _slice(x_train, 0, 0),
            _slice(y_train, 0, 0),
            0,
            "train_tail_fraction_fallback_no_calibration",
        )

    # Invariant: calibration set size must be > 0 and fit set size must be > 0
    # to be considered a valid calibration split.
    desired_cal_size = max(1, int(n_samples * validation_fraction))
    # Preserve at least 2 samples for fit data to avoid trivial failures.
    cal_size = min(desired_cal_size, max(0, n_samples - 2))

    # Keep at least two classes in fit slice when possible to avoid
    # single-class XGBoost fit failures on tiny datasets.
    # We search for the first index where the label differs from the first label
    # to find the minimum fit size that has at least two classes.
    if cal_size > 0 and n_samples > 1:
        y_fit_candidate_full = _slice(y_train, 0, n_samples - cal_size)
        if _nunique(y_fit_candidate_full) < 2:
            # Need to reduce cal_size (increase fit size) until we have 2 classes.
            first_val = y_train.iloc[0] if hasattr(y_train, "iloc") else y_train[0]
            # Find first index i such that y_train[i] != first_val
            # This is O(N) and much faster than the iterative nunique approach.
            found_idx = -1
            for i in range(1, n_samples):
                val = y_train.iloc[i] if hasattr(y_train, "iloc") else y_train[i]
                if val != first_val:
                    found_idx = i
                    break

            if found_idx != -1:
                # Minimum fit_size needed is found_idx + 1
                min_fit_size = found_idx + 1
                cal_size = min(cal_size, n_samples - min_fit_size)
            else:
                # All labels are the same, cannot get 2 classes.
                cal_size = 0

    # Tighten invariant: if cal_size would be 0 or fit_size would be 0,
    # fallback to full train without calibration.
    fit_size = n_samples - cal_size
    if cal_size <= 0 or fit_size <= 0:
        cal_size = 0
        split_strategy = "train_tail_fraction_fallback_no_calibration"
    else:
        split_strategy = "train_tail_fraction"

    split_at = n_samples - cal_size
    x_fit = _slice(x_train, 0, split_at)
    y_fit = _slice(y_train, 0, split_at)
    x_cal = _slice(x_train, split_at, n_samples)
    y_cal = _slice(y_train, split_at, n_samples)

    return x_fit, y_fit, x_cal, y_cal, cal_size, split_strategy


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
    feature_set_id: str | None = None,
    feature_resolution_mode: str = "strict",
    feature_groups: list[str] | None = None,
    n_jobs: int | None = None,
    min_cal_samples: int = 100,
    min_cal_pos: int = 10,
    min_cal_neg: int = 10,
) -> str:
    """Train an XGBoost model with MLflow tracking."""
    _mlflow = _get_mlflow()
    _get_xgb()
    _run_tuning = _get_run_tuning_study()
    import numpy as np
    from mlflow.models import infer_signature

    from features.registry import FeatureRegistry
    from training.schemas import SplitStrategy

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

    actual_feature_columns = (
        feature_columns if feature_columns is not None else loader.FEATURE_COLUMNS
    )
    # Validate features against registry
    for f in actual_feature_columns:
        FeatureRegistry.get(f)

    # Create FeatureSetSpec (Commit 8)
    from features.spec import FeatureSetSpec

    feature_spec = FeatureSetSpec.from_features(actual_feature_columns)

    if split.train_size == 0:
        raise ValueError("No training data available. Generate data first.")
    if split.test_size == 0:
        raise ValueError("No test data available. Adjust training_window_days.")

    n_negative = (split.y_train == 0).sum()
    n_positive = (split.y_train == 1).sum()

    # Phase 2.1: Persist feature schema hash
    ordered_features = sorted(actual_feature_columns)
    schema_json = json.dumps(ordered_features)
    feature_schema_hash = hashlib.sha256(schema_json.encode("utf-8")).hexdigest()

    if n_positive == 0:
        cutoff = training_cutoff_date.date()
        raise ValueError(f"No fraud samples in training set (cutoff: {cutoff}).")

    if scale_pos_weight is None:
        scale_pos_weight = n_negative / n_positive

    validation_fraction = split_config.validation_fraction if split_config else 0.2
    (
        x_fit_base,
        y_fit_base,
        x_cal,
        y_cal,
        calibration_set_size,
        calibration_split_strategy,
    ) = _derive_calibration_split(
        split.X_train,
        split.y_train,
        validation_fraction=validation_fraction,
    )
    calibration_fit_strategy = calibration_split_strategy
    if calibration_set_size == 0:
        calibration_fit_strategy = "train_tail_fraction_fallback_full_train"

    import time

    training_start_time = time.time()
    effective_early_stopping_rounds = early_stopping_rounds
    early_stopping_override = None

    with _mlflow.start_run() as run, _mlflow.start_span(name="train_model") as span:
        span.set_attribute("feature_count", len(actual_feature_columns))
        _mlflow.set_tags(
            {
                "git_sha": _get_git_sha() or "unknown",
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "xgboost_version": xgb_pkg.__version__,
            }
        )

        if split.split_manifest is not None:
            _mlflow.set_tags(
                {
                    "split.strategy": split.split_manifest.get("strategy", "unknown"),
                    "split.train_size": str(split.split_manifest.get("train_size", 0)),
                    "split.test_size": str(split.split_manifest.get("test_size", 0)),
                    "split.train_fraud_rate": str(
                        split.split_manifest.get("train_fraud_rate", 0.0)
                    ),
                    "split.test_fraud_rate": str(
                        split.split_manifest.get("test_fraud_rate", 0.0)
                    ),
                }
            )

        trials_df = None
        if (
            tuning_config is not None
            and tuning_config.enabled
            and len(y_fit_base) >= 10
        ):
            n = len(y_fit_base)
            val_size = max(1, int(n * validation_fraction))
            train_size = n - val_size
            if train_size >= 10:
                x_tr = x_fit_base.iloc[:train_size]
                y_tr = y_fit_base.iloc[:train_size]
                x_val = x_fit_base.iloc[train_size:]
                y_val = y_fit_base.iloc[train_size:]

                if early_stopping_rounds is not None:
                    val_count = len(y_val)
                    tune_val_size = val_count // 2
                    es_val_size = val_count - tune_val_size
                    _mlflow.log_params(
                        {
                            "tuning_val_size": int(tune_val_size),
                            "early_stop_val_size": int(es_val_size),
                        }
                    )

                    if tune_val_size >= 1 and es_val_size >= 1:
                        x_tune_val = x_val.iloc[:tune_val_size]
                        y_tune_val = y_val.iloc[:tune_val_size]
                        x_es_val = x_val.iloc[tune_val_size:]
                        y_es_val = y_val.iloc[tune_val_size:]
                        early_stopping_override = (
                            train_size + tune_val_size,
                            x_es_val,
                            y_es_val,
                        )
                        x_val = x_tune_val
                        y_val = y_tune_val
                        _mlflow.set_tag(
                            "tuning_es_split_policy", "first_half_tune_second_half_es"
                        )
                    else:
                        effective_early_stopping_rounds = None
                        disabled_reason = "validation_slice_too_small"
                        _mlflow.log_param(
                            "early_stopping_disabled_reason", disabled_reason
                        )
                        logger.warning(
                            f"Early stopping disabled (reason: {disabled_reason}): "
                            f"requested {early_stopping_rounds} rounds, but validation "
                            f"slice (size={val_count}) too small to split disjointly."
                        )

                best, trials_df = _run_tuning(
                    x_tr,
                    y_tr,
                    x_val,
                    y_val,
                    n_trials=tuning_config.n_trials,
                    metric=tuning_config.metric,
                    timeout_seconds=tuning_config.timeout_minutes * 60,
                    seed=split_config.seed if split_config else 42,
                    scale_pos_weight=scale_pos_weight,
                    direction=tuning_config.direction,
                    strategy=tuning_config.strategy.value,
                    search_space_overrides=tuning_config.search_space,
                )
                selected_params = best
                selection_type = "auto"
                selected_trial_num = None
                if tuning_config.selected_trial_number is not None:
                    from model.tuning import get_trial_params

                    manual = get_trial_params(
                        trials_df, tuning_config.selected_trial_number
                    )
                    if manual:
                        selected_params = manual
                        selection_type = "manual"
                        selected_trial_num = tuning_config.selected_trial_number
                if selected_params:
                    max_depth = selected_params.get("max_depth", max_depth)
                    n_estimators = selected_params.get("n_estimators", n_estimators)
                    learning_rate = selected_params.get("learning_rate", learning_rate)
                    min_child_weight = selected_params.get(
                        "min_child_weight", min_child_weight
                    )
                    subsample = selected_params.get("subsample", subsample)
                    colsample_bytree = selected_params.get(
                        "colsample_bytree", colsample_bytree
                    )
                    gamma = selected_params.get("gamma", gamma)
                    reg_alpha = selected_params.get("reg_alpha", reg_alpha)
                    reg_lambda = selected_params.get("reg_lambda", reg_lambda)
                    for k, v in selected_params.items():
                        _mlflow.log_param(f"tuning_best_{k}", v)

                _mlflow.set_tags(
                    {
                        "tuning.selected_trial": str(selected_trial_num)
                        if selected_trial_num is not None
                        else "best",
                        "tuning.selection_type": selection_type,
                        "tuning.n_trials": str(tuning_config.n_trials),
                    }
                )

        # ... after param selection ...
        final_hyperparams = {
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
            "early_stopping_rounds": early_stopping_rounds,
        }

        # Compute Unified Training Config Hash (Commit 4)
        config_to_hash = {
            "features": sorted(actual_feature_columns),
            "hyperparameters": final_hyperparams,
            "split_config": (split_config.model_dump() if split_config else None),
            "training_window_days": training_window_days,
        }
        # Ensure deterministic JSON
        config_json = json.dumps(
            config_to_hash, sort_keys=True, default=_to_python_type
        )
        training_config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        _mlflow.set_tag("training_config_hash", training_config_hash)
        _mlflow.log_dict(config_to_hash, "resolved_training_config.json")

        # Log feature set spec (Commit 8)
        _mlflow.log_dict(feature_spec.model_dump(), "feature_set.json")
        _mlflow.set_tag("feature_set_hash", feature_spec.hash)
        _mlflow.set_tag("feature_schema_hash", feature_schema_hash)
        _mlflow.log_dict(
            {
                "feature_schema_hash": feature_schema_hash,
                "feature_count": len(ordered_features),
            },
            "feature_schema_hash.json",
        )

        # Log training run spec (NF1)
        from training.schemas import TrainingRunSpec

        run_id = _safe_run_id(run, _mlflow)
        run_spec = TrainingRunSpec(
            run_id=run_id,
            model_name=EXPERIMENT_NAME,
            created_at=datetime.now(UTC).isoformat(),
            training_config_hash=training_config_hash,
            feature_set_id=feature_set_id,
            feature_set_hash=feature_spec.hash,
            resolved_features=actual_feature_columns,
            feature_resolution_mode=feature_resolution_mode,
            requested_feature_groups=feature_groups,
            split_config=split_config,
            tuning_config=tuning_config,
            training_window_days=training_window_days,
        )
        _mlflow.log_dict(run_spec.model_dump(), "training_run_spec.json")
        _mlflow.set_tag("training_run_spec_version", str(run_spec.schema_version))

        params_log = {
            "scale_pos_weight": scale_pos_weight,
            "max_depth": max_depth,
            "training_window_days": training_window_days,
            "train_size": split.train_size,
            "test_size": split.test_size,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "feature_columns": json.dumps(actual_feature_columns),
            "calibration_set_size": int(calibration_set_size),
            "train_fit_set_size": int(len(y_fit_base)),
            "calibration_fraction_effective": (
                float(calibration_set_size) / split.train_size
                if split.train_size > 0
                else 0.0
            ),
            "calibration_positive_rate": (
                float(y_cal.mean()) if calibration_set_size > 0 else 0.0
            ),
            "calibration_split_strategy": calibration_fit_strategy,
        }
        if early_stopping_rounds:
            params_log["early_stopping_rounds"] = early_stopping_rounds
        _mlflow.log_params(params_log)

        clf_kw = {
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
            "n_jobs": n_jobs,
        }
        if effective_early_stopping_rounds:
            clf_kw["early_stopping_rounds"] = effective_early_stopping_rounds
        clf = XGBClassifier(**clf_kw)

        # CV loop
        do_cv = split_config and split_config.strategy == SplitStrategy.KFOLD_TEMPORAL
        if do_cv and len(y_fit_base) >= split_config.n_folds:
            k = split_config.n_folds
            n = len(y_fit_base)
            fold_metrics = []
            x_tr = x_fit_base
            y_tr = y_fit_base
            fold_size = n // k
            for fold_i in range(k):
                val_start = fold_i * fold_size
                val_end = n if fold_i == k - 1 else (fold_i + 1) * fold_size
                val_idx = np.arange(val_start, val_end)
                train_idx = np.concatenate(
                    [np.arange(0, val_start), np.arange(val_end, n)]
                )
                if not len(train_idx) or not len(val_idx):
                    continue
                fold_clf = XGBClassifier(**clf_kw)
                fold_clf.fit(x_tr.iloc[train_idx], y_tr.iloc[train_idx])
                y_vp = fold_clf.predict(x_tr.iloc[val_idx])
                y_vprob = fold_clf.predict_proba(x_tr.iloc[val_idx])[:, 1]
                fm = _compute_metrics(y_tr.iloc[val_idx], y_vp, y_vprob)
                fold_metrics.append(fm)
                for key, val in fm.items():
                    if val is not None:
                        _mlflow.log_metric(f"cv_{key}_fold_{fold_i}", val, step=fold_i)
                    else:
                        logger.warning(f"Fold {fold_i} is degenerate for metric {key}")
            if fold_metrics:
                agg = {}
                for key in fold_metrics[0].keys():
                    vals = [m[key] for m in fold_metrics if m[key] is not None]
                    if vals:
                        agg[f"{key}_mean"] = float(np.mean(vals))
                        agg[f"{key}_std"] = float(np.std(vals))
                        agg[f"{key}_min"] = float(np.min(vals))
                        agg[f"{key}_max"] = float(np.max(vals))
                        agg[f"cv_valid_folds_{key}"] = float(len(vals))
                    else:
                        logger.warning(f"No valid folds for CV metric: {key}")
                _mlflow.log_metrics(agg)
                _mlflow.set_tags({"cv.enabled": "true", "cv.n_folds": str(k)})

        x_fit, y_fit = x_fit_base, y_fit_base
        if effective_early_stopping_rounds is not None and len(y_fit_base) >= 20:
            if early_stopping_override is not None:
                fit_end, x_es_val, y_es_val = early_stopping_override
                x_fit = x_fit_base.iloc[:fit_end]
                y_fit = y_fit_base.iloc[:fit_end]
                clf.fit(x_fit, y_fit, eval_set=[(x_es_val, y_es_val)])
                if hasattr(clf, "best_iteration") and clf.best_iteration is not None:
                    _mlflow.log_metric("best_iteration", int(clf.best_iteration))
            else:
                n = len(y_fit_base)
                val_size = max(1, int(n * validation_fraction))
                train_size = n - val_size
                if train_size >= 10 and val_size >= 1:
                    x_fit = x_fit_base.iloc[:train_size]
                    y_fit = y_fit_base.iloc[:train_size]
                    x_val = x_fit_base.iloc[train_size:]
                    y_val = y_fit_base.iloc[train_size:]
                    clf.fit(x_fit, y_fit, eval_set=[(x_val, y_val)])
                    if (
                        hasattr(clf, "best_iteration")
                        and clf.best_iteration is not None
                    ):
                        _mlflow.log_metric("best_iteration", int(clf.best_iteration))
                else:
                    clf.fit(x_fit, y_fit)
        else:
            clf.fit(x_fit, y_fit)

        y_pred = clf.predict(split.X_test)
        y_pred_proba = clf.predict_proba(split.X_test)[:, 1]
        metrics_dict = _compute_metrics(split.y_test, y_pred, y_pred_proba)
        _mlflow.log_metrics(metrics_dict)

        # Fit calibrator on dedicated calibration slice from training data.
        from model.evaluate import ScoreCalibrator

        x_cal_fit = x_cal
        y_cal_fit = y_cal
        if calibration_set_size == 0:
            x_cal_fit = x_fit_base
            y_cal_fit = y_fit_base

        # Phase 1.1: Calibration viability guards
        cal_skip_reason = None
        n_cal_samples = len(y_cal_fit)
        n_cal_pos = int(np.sum(y_cal_fit == 1))
        n_cal_neg = int(np.sum(y_cal_fit == 0))

        if n_cal_samples < min_cal_samples:
            cal_skip_reason = "calibration_skipped_insufficient_samples"
        elif n_cal_pos < min_cal_pos:
            cal_skip_reason = "calibration_skipped_insufficient_positives"
        elif n_cal_neg < min_cal_neg:
            cal_skip_reason = "calibration_skipped_insufficient_negatives"

        calibrator = ScoreCalibrator()
        cal_params = {
            "calibration_samples": n_cal_samples,
            "calibration_pos_count": n_cal_pos,
            "calibration_neg_count": n_cal_neg,
        }

        if cal_skip_reason:
            logger.warning(
                "Skipping calibration reason_code=%s cal_size=%d "
                "pos_count=%d neg_count=%d",
                cal_skip_reason,
                n_cal_samples,
                n_cal_pos,
                n_cal_neg,
            )
            cal_params.update(
                {
                    "calibration_enabled": False,
                    "calibration_skip_reason": cal_skip_reason,
                }
            )
        else:
            y_cal_proba = np.asarray(clf.predict_proba(x_cal_fit)[:, 1])
            if len(y_cal_proba) != len(y_cal_fit):
                # Defensive alignment for mocked/non-standard predict_proba outputs.
                y_cal_proba = y_cal_proba[: len(y_cal_fit)]
            calibrator.fit(y_cal_proba, y_cal_fit)
            cal_params["calibration_enabled"] = True

        _mlflow.log_params(cal_params)

        signature = infer_signature(split.X_train, y_pred_proba)
        _mlflow.sklearn.log_model(
            clf, "model", signature=signature, input_example=split.X_train.iloc[:1]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, "reference_data.parquet")
            split.X_test.to_parquet(ref_path, index=False)
            _mlflow.log_artifact(ref_path)

            # Save calibrator artifact (C2)
            calibrator_path = os.path.join(tmpdir, "calibrator.pkl")
            import joblib

            joblib.dump(calibrator, calibrator_path)
            _mlflow.log_artifact(calibrator_path)

            # Save score distribution baseline (C3)
            calibrated_scores = calibrator.transform(y_pred_proba)
            buckets = [1, 11, 31, 71, 91, 100]
            counts, _ = np.histogram(calibrated_scores, bins=buckets)
            ratios = counts / len(calibrated_scores)
            baseline_dist = {
                "buckets": [
                    [buckets[i], buckets[i + 1] - 1] for i in range(len(buckets) - 1)
                ],
                "ratios": ratios.tolist(),
                "counts": counts.tolist(),
                "total": int(len(calibrated_scores)),
            }
            dist_path = os.path.join(tmpdir, "score_distribution.json")
            with open(dist_path, "w") as f:
                json.dump(baseline_dist, f, indent=2)
            _mlflow.log_artifact(dist_path)

            # Save feature columns list as artifact for inference (Legacy)
            feature_columns_path = os.path.join(tmpdir, "feature_columns.json")
            with open(feature_columns_path, "w") as f:
                json.dump(actual_feature_columns, f, indent=2)
            _mlflow.log_artifact(feature_columns_path)

            # Save required_features.json artifact (FF5)
            required_features_data = {
                "features": actual_feature_columns,
                "feature_set_hash": feature_spec.hash,
                "training_config_hash": training_config_hash,
            }
            required_features_path = os.path.join(tmpdir, "required_features.json")
            with open(required_features_path, "w") as f:
                json.dump(required_features_data, f, indent=2)
            _mlflow.log_artifact(required_features_path)

            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            _save_confusion_matrix_plot(split.y_test, y_pred, cm_path)
            _mlflow.log_artifact(cm_path)

            fi_base = os.path.join(tmpdir, "feature_importance")
            fi_json, fi_png = _save_feature_importance(
                clf, actual_feature_columns, fi_base
            )
            _mlflow.log_artifact(fi_json)
            _mlflow.log_artifact(fi_png)

            model_card_path = os.path.join(tmpdir, "model_card.md")
            model_card_params = {
                **params_log,
                "train_fraud_rate": float((split.y_train == 1).mean()),
                "test_fraud_rate": float((split.y_test == 1).mean()),
            }
            _generate_model_card(model_card_params, metrics_dict, model_card_path)
            _mlflow.log_artifact(model_card_path)

        _mlflow.log_metric("training_time_seconds", time.time() - training_start_time)
        model_uri = f"runs:/{run_id}/model"
        _mlflow.register_model(model_uri, EXPERIMENT_NAME)
        return run_id


def get_latest_model_version(model_name: str = EXPERIMENT_NAME) -> int | None:
    """Get the latest version number of a registered model."""
    _mlflow = _get_mlflow()
    client = _mlflow.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            return max(int(v.version) for v in versions)
    except Exception:
        pass
    return None


def load_production_model(model_name: str = EXPERIMENT_NAME):
    """Load the latest version of the production model."""
    _mlflow = _get_mlflow()
    import mlflow.sklearn

    version = get_latest_model_version(model_name)
    if version is None:
        raise ValueError(f"No model versions found for '{model_name}'")
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.sklearn.load_model(model_uri)
