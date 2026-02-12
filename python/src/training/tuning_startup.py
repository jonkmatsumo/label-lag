from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

from training.config import TrainingServerConfig
from training.job_queue import JobQueue
from training.job_store import InMemoryJobStore, JobStore
from training.jobs import TuningJobStatus
from training.postgres_job_store import PostgresJobStore

logger = logging.getLogger(__name__)

STALE_HEARTBEAT_ERROR = "worker restart / stale heartbeat"


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

        def mark_stale_failed(j):
            j.status = TuningJobStatus.FAILED
            j.error_message = STALE_HEARTBEAT_ERROR
            j.updated_at = reference_time
            j.ended_at = reference_time

        job_store.update(job.job_id, mark_stale_failed)
        stale_count += 1

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
