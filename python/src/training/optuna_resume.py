from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import optuna

from training.job_store import JobStore
from training.jobs import TrialRecord

logger = logging.getLogger(__name__)

_STUDY_PREFIX = "tuning_job_"


def normalize_optuna_storage_url(url: str | None) -> str | None:
    if not url:
        return None
    normalized = url.strip()
    if normalized.startswith("postgres://"):
        # Optuna/SQLAlchemy expects the "postgresql://" URL scheme.
        return normalized.replace("postgres://", "postgresql://", 1)
    return normalized


def get_optuna_storage_url() -> str | None:
    # TUNING_OPTUNA_STORAGE_URL overrides study storage. If unset, DATABASE_URL is used.
    configured = os.getenv("TUNING_OPTUNA_STORAGE_URL")
    if configured:
        return normalize_optuna_storage_url(configured)
    return normalize_optuna_storage_url(os.getenv("DATABASE_URL"))


def get_tuning_study_name(job_id: str) -> str:
    return f"{_STUDY_PREFIX}{job_id}"


def create_tuning_study(
    *,
    direction: str,
    sampler: optuna.samplers.BaseSampler,
    pruner: optuna.pruners.BasePruner,
    job_id: str | None = None,
    storage_url: str | None = None,
) -> optuna.study.Study:
    resolved_storage = (
        normalize_optuna_storage_url(storage_url)
        if storage_url is not None
        else get_optuna_storage_url()
    )
    study_name = get_tuning_study_name(job_id) if job_id else None

    if resolved_storage and study_name:
        return optuna.create_study(
            study_name=study_name,
            storage=resolved_storage,
            load_if_exists=True,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

    if resolved_storage:
        return optuna.create_study(
            storage=resolved_storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

    return optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)


def load_existing_tuning_study(
    *,
    job_id: str,
    direction: str,
    storage_url: str | None = None,
) -> optuna.study.Study | None:
    resolved_storage = (
        normalize_optuna_storage_url(storage_url)
        if storage_url is not None
        else get_optuna_storage_url()
    )
    if not resolved_storage:
        return None

    try:
        return optuna.load_study(
            study_name=get_tuning_study_name(job_id),
            storage=resolved_storage,
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
        )
    except KeyError:
        return None
    except Exception as exc:
        logger.warning("Unable to load study for job %s: %s", job_id, exc)
        return None


def study_can_resume(study: optuna.study.Study, total_trials: int) -> bool:
    completed_trials = sum(
        1 for trial in study.trials if str(trial.state) == "TrialState.COMPLETE"
    )
    return completed_trials < max(0, total_trials)


def sync_job_from_optuna(
    *,
    job_store: JobStore,
    job_id: str,
    study: optuna.study.Study,
    now: datetime | None = None,
) -> dict[str, int]:
    completed_trials = 0
    pruned_trials = 0
    for trial in study.trials:
        state = str(trial.state)
        if state == "TrialState.COMPLETE":
            completed_trials += 1
        elif state == "TrialState.PRUNED":
            pruned_trials += 1

        job_store.append_trial(
            job_id,
            TrialRecord(
                trial_number=trial.number,
                state=state,
                value=float(trial.value) if trial.value is not None else None,
                params={k: str(v) for k, v in trial.params.items()},
                started_at=trial.datetime_start,
                ended_at=trial.datetime_complete,
                duration_ms=(
                    (trial.datetime_complete - trial.datetime_start).total_seconds()
                    * 1000
                    if trial.datetime_complete and trial.datetime_start
                    else None
                ),
            ),
        )

    best_value: float | None = None
    best_params: dict[str, str] = {}
    try:
        best_trial = study.best_trial
        if best_trial is not None:
            best_value = (
                float(study.best_value) if study.best_value is not None else None
            )
            best_params = {k: str(v) for k, v in study.best_params.items()}
    except Exception:
        best_value = None
        best_params = {}

    update_ts = now if now else datetime.now(UTC)

    def update_job(job):
        job.completed_trials = completed_trials
        job.pruned_trials = pruned_trials
        job.best_value = best_value
        job.best_params = best_params
        job.updated_at = update_ts

    job_store.update(job_id, update_job)
    return {
        "completed_trials": completed_trials,
        "pruned_trials": pruned_trials,
    }
