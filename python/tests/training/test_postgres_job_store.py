from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from training.jobs import TuningJob, TuningJobStatus
from training.postgres_job_store import PostgresJobStore


def _cursor_and_connect_mock(monkeypatch: pytest.MonkeyPatch):
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = None

    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = None
    conn.cursor.return_value = cursor

    monkeypatch.setattr("training.postgres_job_store.psycopg2.connect", lambda _: conn)
    return cursor


def _job_row(job_id: str) -> dict:
    now = datetime.now(UTC)
    return {
        "job_id": job_id,
        "status": TuningJobStatus.PENDING.value,
        "config": {"feature_columns": ["a", "b"]},
        "created_at": now,
        "started_at": None,
        "heartbeat_at": None,
        "updated_at": now,
        "ended_at": None,
        "mlflow_run_id": "run-123",
        "requested_by": "tester",
        "total_trials": 10,
        "completed_trials": 0,
        "pruned_trials": 0,
        "best_value": None,
        "best_params": {},
        "error_message": None,
    }


def _sql_calls(cursor: MagicMock) -> list[str]:
    return [call.args[0] for call in cursor.execute.call_args_list]


def test_ensure_tables_exist_executes_expected_schema(monkeypatch: pytest.MonkeyPatch):
    cursor = _cursor_and_connect_mock(monkeypatch)
    PostgresJobStore("postgres://localhost/test")

    statements = " ".join(_sql_calls(cursor)).lower()
    assert "create table if not exists tuning_jobs" in statements
    assert "create table if not exists tuning_trials" in statements
    assert "idx_tuning_jobs_status_updated_at" in statements
    assert "idx_tuning_trials_job_trial_number" in statements


def test_create_persists_job(monkeypatch: pytest.MonkeyPatch):
    cursor = _cursor_and_connect_mock(monkeypatch)
    store = PostgresJobStore("postgres://localhost/test")
    cursor.execute.reset_mock()

    job = TuningJob.create(config={"feature_columns": ["a"]}, total_trials=5)
    store.create(job)

    statements = " ".join(_sql_calls(cursor)).lower()
    assert "insert into tuning_jobs" in statements


def test_update_selects_for_update_and_persists(monkeypatch: pytest.MonkeyPatch):
    cursor = _cursor_and_connect_mock(monkeypatch)
    cursor.fetchone.return_value = _job_row("job-1")
    cursor.fetchall.return_value = []
    store = PostgresJobStore("postgres://localhost/test")
    cursor.execute.reset_mock()

    def mutate(job):
        job.status = TuningJobStatus.RUNNING
        job.completed_trials = 2

    updated = store.update("job-1", mutate)

    statements = " ".join(_sql_calls(cursor)).lower()
    assert "for update" in statements
    assert "update tuning_jobs" in statements
    assert updated.status == TuningJobStatus.RUNNING
    assert updated.completed_trials == 2


def test_list_trials_trial_number_cursor(monkeypatch: pytest.MonkeyPatch):
    cursor = _cursor_and_connect_mock(monkeypatch)
    cursor.fetchall.return_value = [
        {
            "trial_number": 3,
            "state": "TrialState.COMPLETE",
            "value": 0.85,
            "params_json": {"max_depth": "6"},
            "started_at": datetime.now(UTC),
            "ended_at": datetime.now(UTC),
            "duration_ms": 120.0,
        },
        {
            "trial_number": 4,
            "state": "TrialState.COMPLETE",
            "value": 0.86,
            "params_json": {"max_depth": "8"},
            "started_at": datetime.now(UTC),
            "ended_at": datetime.now(UTC),
            "duration_ms": 121.0,
        },
    ]
    store = PostgresJobStore("postgres://localhost/test")
    cursor.execute.reset_mock()

    trials, next_cursor = store.list_trials("job-1", limit=1, cursor="2")

    statements = " ".join(_sql_calls(cursor)).lower()
    assert "trial_number > %s" in statements
    assert len(trials) == 1
    assert next_cursor == "3"


def test_set_heartbeat_updates_job(monkeypatch: pytest.MonkeyPatch):
    cursor = _cursor_and_connect_mock(monkeypatch)
    cursor.rowcount = 1
    store = PostgresJobStore("postgres://localhost/test")
    cursor.execute.reset_mock()

    store.set_heartbeat("job-1", datetime.now(UTC))

    statements = " ".join(_sql_calls(cursor)).lower()
    assert "set heartbeat_at = %s, updated_at = %s" in statements
