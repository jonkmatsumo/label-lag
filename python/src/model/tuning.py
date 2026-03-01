"""Optuna-based hyperparameter tuning for XGBoost."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import optuna
import pandas as pd

if TYPE_CHECKING:
    from training.job_store import JobStore

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

from training.optuna_resume import create_tuning_study
from training.reason_codes import (
    MLFLOW_TAG_BEST_PARAMS_JSON,
    MLFLOW_TAG_BEST_TRIAL_NUMBER,
    MLFLOW_TAG_TUNING_RESUME_REASON,
    ResumeValidationReason,
)

logger = logging.getLogger(__name__)

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
MAX_BEST_PARAMS_TAG_LENGTH = 2000
MAX_DATASET_IDENTITY_LENGTH = 128
OPTUNA_OBJECTIVE_VERSION = "xgb_objective_v2"
_LEGACY_STUDY_WARNED = False
mlflow: Any = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _set_resume_validation_reason_tag(reason_code: str) -> None:
    """Best-effort MLflow tag for resume validation reason code."""
    if _active_mlflow_run_id() is None:
        return
    try:
        mlflow.set_tag(MLFLOW_TAG_TUNING_RESUME_REASON, reason_code)
    except Exception:
        pass


def _active_mlflow_run_id() -> str | None:
    """Return active MLflow run_id when available, otherwise None."""
    if mlflow is None:
        return None
    active_run = getattr(mlflow, "active_run", None)
    if not callable(active_run):
        return None
    try:
        run = active_run()
    except Exception:
        return None
    run_id = getattr(getattr(run, "info", None), "run_id", None)
    if isinstance(run_id, str) and run_id.strip():
        return run_id
    return None


def _is_transient_mlflow_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "timeout",
            "temporarily",
            "connection",
            "unavailable",
            "connection reset",
            "429",
            "503",
        )
    )


def _best_params_tag_value(best_params: dict[str, object]) -> str:
    serialized = json.dumps(
        {str(key): str(value) for key, value in best_params.items()},
        sort_keys=True,
    )
    return serialized[:MAX_BEST_PARAMS_TAG_LENGTH]


def _coerce_optional_dict(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        try:
            value = payload.model_dump()
            if isinstance(value, dict):
                return value
        except Exception:
            return None
    return None


def _compute_split_config_hash(split_config: Any) -> str | None:
    split_payload = _coerce_optional_dict(split_config)
    if split_payload is None:
        return None

    normalized: dict[str, Any] = {}
    for key in (
        "strategy",
        "n_folds",
        "stratify_column",
        "group_column",
        "validation_fraction",
        "seed",
    ):
        value = split_payload.get(key)
        if value is None or value == "":
            continue
        normalized[key] = value

    if not normalized:
        return None

    serialized = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_dataset_identity(dataset_identity: str | None) -> str | None:
    if dataset_identity is None:
        return None
    normalized = str(dataset_identity).strip()
    if not normalized:
        return None
    return normalized[:MAX_DATASET_IDENTITY_LENGTH]


def _build_study_invariants(
    *,
    split_config: Any,
    dataset_identity: str | None,
    objective_version: str | None,
) -> dict[str, str]:
    invariants: dict[str, str] = {}
    split_hash = _compute_split_config_hash(split_config)
    if split_hash is not None:
        invariants["split_config_hash"] = split_hash

    normalized_dataset_identity = _normalize_dataset_identity(dataset_identity)
    if normalized_dataset_identity is not None:
        invariants["dataset_identity"] = normalized_dataset_identity

    normalized_objective_version = (
        str(objective_version).strip() if objective_version is not None else ""
    )
    if normalized_objective_version:
        invariants["objective_version"] = normalized_objective_version

    return invariants


def _resume_invariant_mismatches(
    *,
    stored_attrs: dict[str, Any],
    expected_attrs: dict[str, str],
) -> list[str]:
    """Identify mismatches between stored and expected study invariants.

    Returns list of mismatch descriptions: 'key (expected=..., actual=...)'
    """
    mismatches: list[str] = []

    def _brief(val: Any) -> str:
        s = str(val).strip()
        if len(s) > 32:
            return f"{s[:16]}...{hashlib.md5(s.encode()).hexdigest()[:8]}"
        return s

    for key, expected_value in expected_attrs.items():
        stored_value = stored_attrs.get(key)
        if stored_value is None:
            mismatches.append(f"{key} (missing)")
            continue
        actual_rendered = str(stored_value).strip()
        if actual_rendered != expected_value:
            mismatches.append(
                f"{key} (expected={_brief(expected_value)}, "
                f"actual={_brief(actual_rendered)})"
            )
    return mismatches


def _create_objective(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str,
    scale_pos_weight: float,
    seed: int,
    search_space: dict,
):
    """Build Optuna objective that trains XGBoost and returns validation metric."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", *search_space["max_depth"]),
            "n_estimators": trial.suggest_int(
                "n_estimators", *search_space["n_estimators"]
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", *search_space["learning_rate"], log=True
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *search_space["min_child_weight"]
            ),
            "subsample": trial.suggest_float("subsample", *search_space["subsample"]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *search_space["colsample_bytree"]
            ),
            "gamma": trial.suggest_float("gamma", *search_space["gamma"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *search_space["reg_alpha"]),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *search_space["reg_lambda"]
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


class JobProgressCallback:
    """Optuna callback to update JobStore with trial results."""

    def __init__(
        self,
        job_id: str,
        job_store: JobStore,
        nested_mlflow_runs: bool | None = None,
    ):
        self.job_id = job_id
        self.job_store = job_store
        # Optional per-trial nested runs. Disabled by default to preserve behavior.
        self._nested_mlflow_runs = (
            _env_flag("TUNING_MLFLOW_NESTED_RUNS", default=False)
            if nested_mlflow_runs is None
            else nested_mlflow_runs
        )

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        from training.jobs import TrialRecord, TuningJobStatus

        record = TrialRecord(
            trial_number=trial.number,
            state=str(trial.state),
            value=float(trial.value) if trial.value is not None else None,
            params={k: str(v) for k, v in trial.params.items()},
            started_at=trial.datetime_start,
            ended_at=trial.datetime_complete,
            duration_ms=(
                (trial.datetime_complete - trial.datetime_start).total_seconds() * 1000
                if trial.datetime_complete and trial.datetime_start
                else None
            ),
        )
        self.job_store.append_trial(self.job_id, record)

        # 1. Update JobStore aggregates
        def update_fn(job):
            # Update counts
            if str(trial.state) == "TrialState.COMPLETE":
                job.completed_trials += 1
            elif str(trial.state) == "TrialState.PRUNED":
                job.pruned_trials += 1

            # Update best value/params
            if study.best_trial and study.best_trial.number == trial.number:
                job.best_value = (
                    float(study.best_value) if study.best_value is not None else None
                )
                job.best_params = {k: str(v) for k, v in study.best_params.items()}

            job.updated_at = datetime.now(UTC)

        self.job_store.update(self.job_id, update_fn)
        self.job_store.set_heartbeat(self.job_id, datetime.now(UTC))
        self._maybe_log_nested_trial_run(trial)

        # 2. Check for cancellation
        job = self.job_store.get(self.job_id)
        if job and job.status == TuningJobStatus.CANCELING:
            logger.info(f"Job {self.job_id} is canceling, stopping study.")
            study.stop()

    def _maybe_log_nested_trial_run(self, trial: optuna.trial.FrozenTrial) -> None:
        if not self._nested_mlflow_runs:
            return
        if str(trial.state) != "TrialState.COMPLETE":
            return

        for attempt in range(2):
            try:
                import mlflow

                with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                    mlflow.set_tag("job_id", self.job_id)
                    mlflow.set_tag("tuning_job_id", self.job_id)
                    mlflow.set_tag("trial_number", str(trial.number))
                    mlflow.set_tag("state", str(trial.state))
                    for key, value in trial.params.items():
                        mlflow.log_param(key, value)
                    if trial.value is not None:
                        mlflow.log_metric("objective_value", float(trial.value))
                return
            except Exception as exc:
                if attempt == 0 and _is_transient_mlflow_error(exc):
                    logger.warning(
                        (
                            "Transient MLflow error for nested run job=%s "
                            "trial=%s; retrying: %s"
                        ),
                        self.job_id,
                        trial.number,
                        exc,
                    )
                    continue
                logger.warning(
                    "Failed to log nested MLflow run for job %s trial=%s: %s",
                    self.job_id,
                    trial.number,
                    exc,
                )
                return


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
    direction: str = "maximize",
    strategy: str = "bayesian",
    search_space_overrides: dict[str, str] | None = None,
    job_id: str | None = None,
    job_store: JobStore | None = None,
    storage_url: str | None = None,
    split_config: dict[str, Any] | None = None,
    dataset_identity: str | None = None,
    objective_version: str = OPTUNA_OBJECTIVE_VERSION,
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
        direction: "maximize" or "minimize".
        strategy: "bayesian", "random", or "grid".
        search_space_overrides: Optional overrides for search space (JSON strings).
        job_id: Optional job ID for tracking.
        job_store: Optional JobStore for tracking.
        storage_url: Optional Optuna RDB storage URL override.
        split_config: Optional split config payload used for data partitioning.
        dataset_identity: Optional bounded dataset identifier for resume checks.
        objective_version: Objective logic version for deterministic resume checks.

    Returns:
        (best_params, trials_df) where trials_df has columns like
        trial, value, params_max_depth, params_learning_rate, ...
    """
    resolved_search_space = DEFAULT_SEARCH_SPACE.copy()
    if search_space_overrides:
        for k, v in search_space_overrides.items():
            if k in resolved_search_space:
                try:
                    resolved_search_space[k] = tuple(json.loads(v))
                except json.JSONDecodeError:
                    pass  # Fallback to default if invalid JSON

    # Log resolved search space
    # (mock mlflow access via module-level placeholder for testability)
    global mlflow
    if mlflow is None:
        try:
            import mlflow as _mlflow

            mlflow = _mlflow
        except ImportError:
            pass

    if mlflow is not None:
        try:
            mlflow.log_dict(resolved_search_space, "resolved_search_space.json")
        except Exception as exc:
            logger.debug("Failed to log search space to MLflow: %s", exc)

    objective = _create_objective(
        x_train,
        y_train,
        x_val,
        y_val,
        metric,
        scale_pos_weight,
        seed,
        resolved_search_space,
    )

    study_invariants = _build_study_invariants(
        split_config=split_config,
        dataset_identity=dataset_identity,
        objective_version=objective_version,
    )

    if strategy == "grid":
        # GridSampler requires search space passed to constructor.
        # Since we use continuous ranges in suggest_*, we need explicit grid values.
        # This requires search space customization (Commit 3).
        pass

    # --- Sub-phase 3.2: recover persisted seed before sampler construction ---
    # If this job already has a study in storage, use its original seed so that
    # resumed runs reproduce the same trial sequence regardless of what the
    # caller passes as `seed`.
    if job_id and (
        storage_url is not None
        or os.getenv("TUNING_OPTUNA_STORAGE_URL")
        or os.getenv("DATABASE_URL")
    ):
        from training.optuna_resume import (
            get_optuna_storage_url,
            get_tuning_study_name,
            normalize_optuna_storage_url,
        )

        resolved = (
            normalize_optuna_storage_url(storage_url)
            if storage_url is not None
            else get_optuna_storage_url()
        )
        if resolved and job_id:
            try:
                existing = optuna.load_study(
                    study_name=get_tuning_study_name(job_id),
                    storage=resolved,
                )
                persisted_seed = existing.user_attrs.get("seed")
                if persisted_seed is not None:
                    seed = int(persisted_seed)
                    logger.info(
                        "Resuming study %s with persisted seed=%d",
                        job_id,
                        seed,
                    )

                invariant_mismatches = _resume_invariant_mismatches(
                    stored_attrs=existing.user_attrs,
                    expected_attrs=study_invariants,
                )
                if invariant_mismatches:
                    # Check if this is a legacy study (missing objective_version)
                    is_legacy = "objective_version" not in existing.user_attrs
                    mismatch_summary = "; ".join(invariant_mismatches)
                    strict_validation = _env_flag(
                        "STRICT_TUNING_RESUME_VALIDATION", default=False
                    )

                    label = (
                        ResumeValidationReason.LEGACY_STUDY.value
                        if is_legacy
                        else ResumeValidationReason.INVARIANT_MISMATCH.value
                    )
                    _set_resume_validation_reason_tag(label)
                    message = (
                        f"{label} (job_id={job_id}; mismatches={mismatch_summary})"
                    )

                    if is_legacy:
                        global _LEGACY_STUDY_WARNED
                        if not _LEGACY_STUDY_WARNED:
                            logger.warning(
                                "%s; legacy study detected without invariants. "
                                "Continuing in warn-only mode (one-time warning).",
                                message,
                            )
                            _LEGACY_STUDY_WARNED = True
                    elif strict_validation:
                        # Use descriptive message with strict-mode disable hint.
                        strict_label = (
                            ResumeValidationReason.INVARIANT_MISMATCH_STRICT.value
                        )
                        _set_resume_validation_reason_tag(strict_label)
                        raise ValueError(
                            f"{strict_label}: {message}. "
                            "Set STRICT_TUNING_RESUME_VALIDATION=0 to allow "
                            "warn-only behavior."
                        )
                    else:
                        logger.warning("%s; continuing in warn-only mode.", message)
            except KeyError:
                pass  # Study doesn't exist yet — first run, use caller seed.
            except ValueError:
                raise
            except Exception:
                pass  # Storage unavailable or lookup error; continue best-effort.

    if strategy == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif strategy == "grid":
        raise ValueError("Grid strategy requires explicit search space definition.")
    else:  # bayesian / default
        sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=5)

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = create_tuning_study(
        direction=direction,
        sampler=sampler,
        pruner=pruner,
        job_id=job_id,
        storage_url=storage_url,
    )
    if job_id:
        study.set_user_attr("tuning_job_id", job_id)
    # Persist seed so resumed studies always reproduce the original sequence.
    if "seed" not in study.user_attrs:
        study.set_user_attr("seed", seed)
    for key, value in study_invariants.items():
        if key not in study.user_attrs:
            study.set_user_attr(key, value)

    existing_trial_count = len(study.trials)
    remaining_trials = max(0, n_trials - existing_trial_count)

    callbacks = []
    if job_id and job_store:
        callbacks.append(JobProgressCallback(job_id, job_store))

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
            callbacks=callbacks,
        )
    best_trial = None
    best = {}
    try:
        best_trial = study.best_trial
        best = study.best_params if best_trial else {}
    except Exception:
        best = {}

    if mlflow is not None:
        try:
            if _active_mlflow_run_id() is not None:
                mlflow.set_tag(
                    MLFLOW_TAG_BEST_TRIAL_NUMBER,
                    str(best_trial.number) if best_trial else "",
                )
                mlflow.set_tag(
                    MLFLOW_TAG_BEST_PARAMS_JSON, _best_params_tag_value(best)
                )
        except Exception as exc:
            logger.warning("Failed to log best trial metadata tags: %s", exc)
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

    # Log trials summary (NF4)
    if mlflow is not None:
        try:
            # Sort trials by value based on direction
            reverse = direction == "maximize"
            completed_trials = [
                t for t in study.trials if str(t.state) == "TrialState.COMPLETE"
            ]
            sorted_trials = sorted(
                completed_trials,
                key=lambda x: (
                    x.value
                    if x.value is not None
                    else (float("-inf") if reverse else float("inf"))
                ),
                reverse=reverse,
            )

            top_k = 10
            summary = {
                "schema_version": 1,
                "metric": metric,
                "direction": direction,
                "total_trials": len(study.trials),
                "completed_trials": len(completed_trials),
                "pruned_trials": pruned_count,
                "top_trials": [
                    {
                        "trial_number": t.number,
                        "value": float(t.value) if t.value is not None else None,
                        "params": t.params,
                        "datetime": (
                            t.datetime_start.isoformat() if t.datetime_start else None
                        ),
                    }
                    for t in sorted_trials[:top_k]
                ],
            }
            mlflow.log_dict(summary, "trials_summary.json")
        except Exception as e:
            logger.warning(f"Failed to log trials summary: {e}")

    return best, trials_df
