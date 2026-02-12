from __future__ import annotations

import logging
import os

import optuna

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
