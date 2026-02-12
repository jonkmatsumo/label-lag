from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from training.jobs import TrialRecord, TuningJob, TuningJobStatus

logger = logging.getLogger(__name__)


def _coerce_utc(ts: datetime | None) -> datetime | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class PostgresJobStore:
    """Durable tuning job storage backed by PostgreSQL."""

    def __init__(self, dsn: str):
        if not dsn:
            raise ValueError("PostgresJobStore requires a non-empty DSN")
        self._dsn = dsn
        self.ensure_tables_exist()

    def _connect(self):
        return psycopg2.connect(self._dsn)

    def ensure_tables_exist(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tuning_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    started_at TIMESTAMPTZ NULL,
                    heartbeat_at TIMESTAMPTZ NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    ended_at TIMESTAMPTZ NULL,
                    mlflow_run_id TEXT NULL,
                    requested_by TEXT NULL,
                    total_trials INTEGER NOT NULL DEFAULT 0,
                    completed_trials INTEGER NOT NULL DEFAULT 0,
                    pruned_trials INTEGER NOT NULL DEFAULT 0,
                    best_value DOUBLE PRECISION NULL,
                    best_params JSONB NOT NULL DEFAULT '{}'::jsonb,
                    error_message TEXT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS tuning_trials (
                    job_id TEXT NOT NULL
                        REFERENCES tuning_jobs(job_id) ON DELETE CASCADE,
                    trial_number INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    value DOUBLE PRECISION NULL,
                    params_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    started_at TIMESTAMPTZ NULL,
                    ended_at TIMESTAMPTZ NULL,
                    duration_ms DOUBLE PRECISION NULL,
                    PRIMARY KEY (job_id, trial_number)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tuning_jobs_status_updated_at
                ON tuning_jobs(status, updated_at DESC)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tuning_trials_job_trial_number
                ON tuning_trials(job_id, trial_number)
                """
            )

    def create(self, job: TuningJob) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tuning_jobs (
                    job_id,
                    status,
                    config,
                    created_at,
                    started_at,
                    heartbeat_at,
                    updated_at,
                    ended_at,
                    mlflow_run_id,
                    requested_by,
                    total_trials,
                    completed_trials,
                    pruned_trials,
                    best_value,
                    best_params,
                    error_message
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    job.job_id,
                    job.status.value,
                    Json(job.config),
                    _coerce_utc(job.created_at) or _utc_now(),
                    _coerce_utc(job.started_at),
                    _coerce_utc(job.heartbeat_at),
                    _coerce_utc(job.updated_at) or _utc_now(),
                    _coerce_utc(job.ended_at),
                    job.mlflow_run_id,
                    job.requested_by,
                    job.total_trials,
                    job.completed_trials,
                    job.pruned_trials,
                    job.best_value,
                    Json(job.best_params),
                    job.error_message,
                ),
            )
            self._upsert_trials(cur, job.job_id, job.trials)

    def get(self, job_id: str) -> TuningJob | None:
        with self._connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM tuning_jobs WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            job = self._row_to_job(row)
            job.trials = self._fetch_trials(cur, job_id)
            return job

    def update(self, job_id: str, mutate_fn: Callable[[TuningJob], None]) -> TuningJob:
        with self._connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM tuning_jobs WHERE job_id = %s FOR UPDATE", (job_id,)
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Job {job_id} not found")
            job = self._row_to_job(row)
            job.trials = self._fetch_trials(cur, job_id)
            mutate_fn(job)
            self._persist_job(cur, job)
            self._upsert_trials(cur, job.job_id, job.trials)
            return job

    def list_jobs(
        self, statuses: list[TuningJobStatus] | None = None, limit: int = 100
    ) -> list[TuningJob]:
        max_limit = max(1, min(limit, 1000))
        with self._connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            if statuses:
                status_values = [status.value for status in statuses]
                cur.execute(
                    """
                    SELECT * FROM tuning_jobs
                    WHERE status = ANY(%s)
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (status_values, max_limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM tuning_jobs
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (max_limit,),
                )
            return [self._row_to_job(row) for row in cur.fetchall()]

    def append_trial(self, job_id: str, trial: TrialRecord) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            self._upsert_trial(cur, job_id, trial)
            cur.execute(
                "UPDATE tuning_jobs SET updated_at = %s WHERE job_id = %s",
                (_utc_now(), job_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Job {job_id} not found")

    def list_trials(
        self,
        job_id: str,
        limit: int = 50,
        cursor: str | None = None,
        sort_by: str = "trial_number",
    ) -> tuple[list[TrialRecord], str | None]:
        max_limit = max(1, min(limit, 200))
        with self._connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            if sort_by == "value":
                cur.execute(
                    """
                    SELECT
                        trial_number,
                        state,
                        value,
                        params_json,
                        started_at,
                        ended_at,
                        duration_ms
                    FROM tuning_trials
                    WHERE job_id = %s
                    ORDER BY value DESC NULLS LAST, trial_number ASC
                    LIMIT %s
                    """,
                    (job_id, max_limit),
                )
                return [self._row_to_trial(row) for row in cur.fetchall()], None

            last_trial = -1
            if cursor:
                try:
                    last_trial = int(cursor)
                except ValueError as exc:
                    raise ValueError(f"Invalid cursor: {cursor}") from exc

            cur.execute(
                """
                SELECT
                    trial_number,
                    state,
                    value,
                    params_json,
                    started_at,
                    ended_at,
                    duration_ms
                FROM tuning_trials
                WHERE job_id = %s AND trial_number > %s
                ORDER BY trial_number ASC
                LIMIT %s
                """,
                (job_id, last_trial, max_limit + 1),
            )
            rows = cur.fetchall()
            has_more = len(rows) > max_limit
            page_rows = rows[:max_limit]
            next_cursor = (
                str(page_rows[-1]["trial_number"]) if page_rows and has_more else None
            )
            return [self._row_to_trial(row) for row in page_rows], next_cursor

    def set_heartbeat(self, job_id: str, ts: datetime) -> None:
        heartbeat_at = _coerce_utc(ts) or _utc_now()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tuning_jobs
                SET heartbeat_at = %s, updated_at = %s
                WHERE job_id = %s
                """,
                (heartbeat_at, heartbeat_at, job_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"Job {job_id} not found")

    def list(self, limit: int = 50) -> list[TuningJob]:
        return self.list_jobs(limit=limit)

    def _persist_job(self, cur, job: TuningJob) -> None:
        cur.execute(
            """
            UPDATE tuning_jobs
            SET
                status = %s,
                config = %s,
                started_at = %s,
                heartbeat_at = %s,
                updated_at = %s,
                ended_at = %s,
                mlflow_run_id = %s,
                requested_by = %s,
                total_trials = %s,
                completed_trials = %s,
                pruned_trials = %s,
                best_value = %s,
                best_params = %s,
                error_message = %s
            WHERE job_id = %s
            """,
            (
                job.status.value,
                Json(job.config),
                _coerce_utc(job.started_at),
                _coerce_utc(job.heartbeat_at),
                _coerce_utc(job.updated_at) or _utc_now(),
                _coerce_utc(job.ended_at),
                job.mlflow_run_id,
                job.requested_by,
                job.total_trials,
                job.completed_trials,
                job.pruned_trials,
                job.best_value,
                Json(job.best_params),
                job.error_message,
                job.job_id,
            ),
        )

    def _upsert_trials(self, cur, job_id: str, trials: Iterable[TrialRecord]) -> None:
        for trial in trials:
            self._upsert_trial(cur, job_id, trial)

    def _upsert_trial(self, cur, job_id: str, trial: TrialRecord) -> None:
        cur.execute(
            """
            INSERT INTO tuning_trials (
                job_id,
                trial_number,
                state,
                value,
                params_json,
                started_at,
                ended_at,
                duration_ms
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (job_id, trial_number) DO UPDATE
            SET
                state = EXCLUDED.state,
                value = EXCLUDED.value,
                params_json = EXCLUDED.params_json,
                started_at = EXCLUDED.started_at,
                ended_at = EXCLUDED.ended_at,
                duration_ms = EXCLUDED.duration_ms
            """,
            (
                job_id,
                trial.trial_number,
                trial.state,
                trial.value,
                Json(trial.params),
                _coerce_utc(trial.started_at),
                _coerce_utc(trial.ended_at),
                trial.duration_ms,
            ),
        )

    def _fetch_trials(self, cur, job_id: str) -> list[TrialRecord]:
        cur.execute(
            """
            SELECT
                trial_number,
                state,
                value,
                params_json,
                started_at,
                ended_at,
                duration_ms
            FROM tuning_trials
            WHERE job_id = %s
            ORDER BY trial_number ASC
            """,
            (job_id,),
        )
        return [self._row_to_trial(row) for row in cur.fetchall()]

    def _row_to_job(self, row: dict) -> TuningJob:
        return TuningJob(
            job_id=row["job_id"],
            config=row["config"] or {},
            status=TuningJobStatus(row["status"]),
            created_at=_coerce_utc(row["created_at"]) or _utc_now(),
            started_at=_coerce_utc(row["started_at"]),
            heartbeat_at=_coerce_utc(row.get("heartbeat_at")),
            updated_at=_coerce_utc(row["updated_at"]) or _utc_now(),
            ended_at=_coerce_utc(row["ended_at"]),
            mlflow_run_id=row["mlflow_run_id"],
            requested_by=row["requested_by"],
            total_trials=row["total_trials"] or 0,
            completed_trials=row["completed_trials"] or 0,
            pruned_trials=row["pruned_trials"] or 0,
            best_value=row["best_value"],
            best_params=row["best_params"] or {},
            error_message=row["error_message"],
        )

    def _row_to_trial(self, row: dict) -> TrialRecord:
        return TrialRecord(
            trial_number=row["trial_number"],
            state=row["state"],
            value=row["value"],
            params=row["params_json"] or {},
            started_at=_coerce_utc(row["started_at"]),
            ended_at=_coerce_utc(row["ended_at"]),
            duration_ms=row["duration_ms"],
        )
