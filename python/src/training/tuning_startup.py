from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

from training.config import TrainingServerConfig
from training.job_queue import JobQueue
from training.job_store import InMemoryJobStore, JobStore
from training.jobs import TuningJobStatus
from training.optuna_resume import (
    load_existing_tuning_study,
    study_can_resume,
    sync_job_from_optuna,
)
from training.postgres_job_store import PostgresJobStore

logger = logging.getLogger(__name__)

STALE_HEARTBEAT_ERROR = "worker restart / stale heartbeat"
STALE_HEARTBEAT_RESUME_UNAVAILABLE_ERROR = (
    "worker restart / stale heartbeat (resume unavailable)"
)


def get_tuning_job_retention_days() -> int:
    # TUNING_JOB_RETENTION_DAYS controls terminal tuning job retention.
    raw = os.getenv("TUNING_JOB_RETENTION_DAYS", "14")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid TUNING_JOB_RETENTION_DAYS=%s; defaulting to 14 days", raw
        )
        return 14


def build_tuning_job_store(config: TrainingServerConfig) -> JobStore:
    configured_mode = os.getenv("TUNING_JOB_STORE")
    default_mode = "postgres" if config.db_dsn else "memory"
    mode = configured_mode.strip().lower() if configured_mode else default_mode

    if mode not in {"memory", "postgres"}:
        logger.warning(
            "Unknown TUNING_JOB_STORE=%s. Falling back to %s.",
            configured_mode,
            default_mode,
        )
        mode = default_mode

    if mode == "postgres":
        if not config.db_dsn:
            if configured_mode:
                raise ValueError(
                    "TUNING_JOB_STORE=postgres requires DATABASE_URL to be set"
                )
            logger.warning("DATABASE_URL missing. Falling back to in-memory JobStore.")
            return InMemoryJobStore()
        return PostgresJobStore(config.db_dsn)

    return InMemoryJobStore()


def reconcile_stale_jobs(
    job_store: JobStore,
    heartbeat_interval_seconds: int,
    now: datetime | None = None,
) -> int:
    reference_time = now if now else datetime.now(UTC)
    stale_cutoff = reference_time - timedelta(seconds=heartbeat_interval_seconds * 2)
    stale_count = 0
    resumed_count = 0

    candidates = job_store.list_jobs(
        statuses=[TuningJobStatus.RUNNING, TuningJobStatus.CANCELING],
        limit=10_000,
    )
    for job in candidates:
        heartbeat_ts = (
            job.heartbeat_at
            or job.updated_at
            or job.started_at
            or job.created_at
            or reference_time
        )
        if heartbeat_ts > stale_cutoff:
            continue

        tuning_cfg = (
            job.config.get("tuning_config", {}) if isinstance(job.config, dict) else {}
        )
        direction = tuning_cfg.get("direction", "maximize")
        study = load_existing_tuning_study(
            job_id=job.job_id,
            direction=direction,
        )

        if study and study_can_resume(study, total_trials=job.total_trials):
            sync_job_from_optuna(
                job_store=job_store,
                job_id=job.job_id,
                study=study,
                now=reference_time,
            )

            def mark_pending(j):
                j.status = TuningJobStatus.PENDING
                j.updated_at = reference_time
                j.ended_at = None
                j.error_message = None

            job_store.update(job.job_id, mark_pending)
            resumed_count += 1
            continue

        def mark_stale_failed(j):
            j.status = TuningJobStatus.FAILED
            j.error_message = STALE_HEARTBEAT_RESUME_UNAVAILABLE_ERROR
            j.updated_at = reference_time
            j.ended_at = reference_time

        job_store.update(job.job_id, mark_stale_failed)
        stale_count += 1

    if resumed_count:
        logger.info(
            "Recovered stale tuning jobs for resume: resumed_jobs=%s", resumed_count
        )

    return stale_count


def reenqueue_pending_jobs(job_store: JobStore, job_queue: JobQueue) -> int:
    pending_jobs = job_store.list_jobs(statuses=[TuningJobStatus.PENDING], limit=10_000)
    for job in sorted(pending_jobs, key=lambda current: current.created_at):
        job_queue.enqueue(job.job_id)
        logger.info(
            "tuning_job_reenqueued job_id=%s queue_depth=%s",
            job.job_id,
            job_queue.depth(),
        )
    return len(pending_jobs)


def prune_tuning_jobs(
    job_store: JobStore,
    retention_days: int | None = None,
    now: datetime | None = None,
) -> int:
    retention = (
        retention_days
        if retention_days is not None
        else get_tuning_job_retention_days()
    )
    reference_time = now if now else datetime.now(UTC)
    older_than = reference_time - timedelta(days=retention)
    return job_store.prune_terminal_jobs(older_than=older_than)
