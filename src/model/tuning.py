"""Optuna-based hyperparameter tuning for XGBoost."""

from __future__ import annotations

import optuna
import pandas as pd
from optuna.integration import XGBoostPruningCallback
from optuna.pruners import MedianPruner
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

_METRIC_FNS = {
    "pr_auc": lambda y, p, prob: average_precision_score(y, prob),
    "roc_auc": lambda y, p, prob: roc_auc_score(y, prob) if len(set(y)) > 1 else 0.0,
    "f1": lambda y, p, prob: f1_score(y, p, zero_division=0),
    "precision": lambda y, p, prob: precision_score(y, p, zero_division=0),
    "recall": lambda y, p, prob: recall_score(y, p, zero_division=0),
}

DEFAULT_SEARCH_SPACE = {
    "max_depth": (2, 12),
    "n_estimators": (50, 300),
    "learning_rate": (0.01, 0.3),
    "min_child_weight": (1, 10),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "gamma": (0.0, 5.0),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.0, 10.0),
}


def _create_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str,
    scale_pos_weight: float,
    seed: int,
):
    """Build Optuna objective that trains XGBoost and returns validation metric."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int(
                "max_depth", *DEFAULT_SEARCH_SPACE["max_depth"]
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", *DEFAULT_SEARCH_SPACE["n_estimators"]
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *DEFAULT_SEARCH_SPACE["learning_rate"], log=True
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *DEFAULT_SEARCH_SPACE["min_child_weight"]
            ),
            "subsample": trial.suggest_float(
                "subsample", *DEFAULT_SEARCH_SPACE["subsample"]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *DEFAULT_SEARCH_SPACE["colsample_bytree"]
            ),
            "gamma": trial.suggest_float("gamma", *DEFAULT_SEARCH_SPACE["gamma"]),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", *DEFAULT_SEARCH_SPACE["reg_alpha"]
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *DEFAULT_SEARCH_SPACE["reg_lambda"]
            ),
        }
        pruning_callback = XGBoostPruningCallback(trial, "validation_0-logloss")
        clf = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            callbacks=[pruning_callback],
            **params,
        )
        clf.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        y_prob = clf.predict_proba(x_val)[:, 1]
        y_pred = clf.predict(x_val)
        fn = _METRIC_FNS.get(metric, _METRIC_FNS["pr_auc"])
        return float(fn(y_val, y_pred, y_prob))

    return objective


def get_trial_params(trials_df: pd.DataFrame, trial_number: int) -> dict:
    """Extract hyperparameters for a specific trial from trials DataFrame.

    Args:
        trials_df: DataFrame with trial history (from run_tuning_study).
        trial_number: Trial number to extract.

    Returns:
        Dictionary of hyperparameters for the specified trial.
        Empty dict if trial not found.

    Raises:
        ValueError: If trial_number is negative.
    """
    if trial_number < 0:
        raise ValueError(f"trial_number must be non-negative, got {trial_number}")

    trial_row = trials_df[trials_df["trial"] == trial_number]
    if trial_row.empty:
        return {}

    params = {}
    for col in trial_row.columns:
        if col.startswith("params_"):
            param_name = col[7:]  # Remove "params_" prefix
            params[param_name] = trial_row[col].iloc[0]
    return params


def run_tuning_study(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 20,
    metric: str = "pr_auc",
    timeout_seconds: int | None = None,
    seed: int = 42,
    scale_pos_weight: float = 1.0,
) -> tuple[dict, pd.DataFrame]:
    """Run Optuna study and return best params and trial history.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_val: Validation features.
        y_val: Validation labels.
        n_trials: Number of Optuna trials.
        metric: Metric to maximize (e.g. pr_auc, roc_auc, f1).
        timeout_seconds: Optional timeout for the study.
        seed: Random seed.
        scale_pos_weight: Class weight for positive class.

    Returns:
        (best_params, trials_df) where trials_df has columns like
        trial, value, params_max_depth, params_learning_rate, ...
    """
    objective = _create_objective(
        x_train, y_train, x_val, y_val, metric, scale_pos_weight, seed
    )
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        show_progress_bar=False,
    )
    best = study.best_params if study.best_trial else {}
    rows = []
    pruned_count = 0
    completed_count = 0
    for t in study.trials:
        row = {
            "trial": t.number,
            "value": t.value if t.value is not None else None,
            "state": str(t.state),
        }
        # Capture pruning step if available
        if hasattr(t, "system_attrs") and "pruned_at_step" in t.system_attrs:
            row["pruned_at_step"] = t.system_attrs["pruned_at_step"]
        else:
            row["pruned_at_step"] = None
        if t.params:
            for k, v in t.params.items():
                row[f"params_{k}"] = v
        rows.append(row)
        if str(t.state) == "TrialState.PRUNED":
            pruned_count += 1
        elif str(t.state) == "TrialState.COMPLETE":
            completed_count += 1
    trials_df = pd.DataFrame(rows)
    # Store pruning stats for logging
    trials_df.attrs = {"pruned_count": pruned_count, "completed_count": completed_count}
    return best, trials_df
