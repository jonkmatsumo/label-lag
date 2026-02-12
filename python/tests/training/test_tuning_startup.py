from __future__ import annotations

from datetime import UTC, datetime, timedelta

from training.config import TrainingServerConfig
from training.job_queue import JobQueue
from training.job_store import InMemoryJobStore
from training.jobs import TuningJob, TuningJobStatus
from training.tuning_startup import (
    STALE_HEARTBEAT_ERROR,
    build_tuning_job_store,
    reconcile_stale_jobs,
    reenqueue_pending_jobs,
)


def test_build_tuning_job_store_defaults_to_memory_without_db(monkeypatch):
    monkeypatch.delenv("TUNING_JOB_STORE", raising=False)
    config = TrainingServerConfig(
        host="0.0.0.0",
        port=50053,
        max_workers=5,
        db_dsn=None,
        mlflow_tracking_uri=None,
    )

    store = build_tuning_job_store(config)

    assert isinstance(store, InMemoryJobStore)


def test_build_tuning_job_store_uses_postgres_when_db_configured(monkeypatch):
    class FakePostgresStore:
        def __init__(self, dsn):
            self.dsn = dsn

    monkeypatch.delenv("TUNING_JOB_STORE", raising=False)
    monkeypatch.setattr("training.tuning_startup.PostgresJobStore", FakePostgresStore)
    config = TrainingServerConfig(
        host="0.0.0.0",
        port=50053,
        max_workers=5,
        db_dsn="postgres://db",
        mlflow_tracking_uri=None,
    )

    store = build_tuning_job_store(config)

    assert isinstance(store, FakePostgresStore)
    assert store.dsn == "postgres://db"


def test_reconcile_stale_jobs_marks_stale_running_as_failed():
    now = datetime.now(UTC)
    store = InMemoryJobStore()

    stale_job = TuningJob.create(config={}, total_trials=5)
    stale_job.status = TuningJobStatus.RUNNING
    stale_job.heartbeat_at = now - timedelta(seconds=30)
    stale_job.updated_at = stale_job.heartbeat_at
    store.create(stale_job)

    fresh_job = TuningJob.create(config={}, total_trials=5)
    fresh_job.status = TuningJobStatus.RUNNING
    fresh_job.heartbeat_at = now - timedelta(seconds=2)
    fresh_job.updated_at = fresh_job.heartbeat_at
    store.create(fresh_job)

    changed = reconcile_stale_jobs(store, heartbeat_interval_seconds=5, now=now)

    assert changed == 1
    assert store.get(stale_job.job_id).status == TuningJobStatus.FAILED
    assert store.get(stale_job.job_id).error_message == STALE_HEARTBEAT_ERROR
    assert store.get(fresh_job.job_id).status == TuningJobStatus.RUNNING


def test_reenqueue_pending_jobs_preserves_fifo_by_created_time():
    store = InMemoryJobStore()
    queue = JobQueue()

    older = TuningJob.create(config={}, total_trials=1)
    newer = TuningJob.create(config={}, total_trials=1)
    older.created_at = datetime.now(UTC) - timedelta(minutes=1)
    newer.created_at = datetime.now(UTC)

    store.create(newer)
    store.create(older)

    count = reenqueue_pending_jobs(store, queue)

    assert count == 2
    assert queue.get(block=False) == older.job_id
    assert queue.get(block=False) == newer.job_id
