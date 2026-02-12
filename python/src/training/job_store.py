from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from training.jobs import TrialRecord, TuningJob, TuningJobStatus


class JobStore(Protocol):
    def create(self, job: TuningJob) -> None: ...

    def get(self, job_id: str) -> TuningJob | None: ...

    def update(
        self, job_id: str, mutate_fn: Callable[[TuningJob], None]
    ) -> TuningJob: ...

    def list_jobs(
        self, statuses: list[TuningJobStatus] | None = None, limit: int = 100
    ) -> list[TuningJob]: ...

    def append_trial(self, job_id: str, trial: TrialRecord) -> None: ...

    def list_trials(
        self,
        job_id: str,
        limit: int = 50,
        cursor: str | None = None,
        sort_by: str = "trial_number",
    ) -> tuple[list[TrialRecord], str | None]: ...

    def set_heartbeat(self, job_id: str, ts: datetime) -> None: ...

    def list(self, limit: int = 50) -> list[TuningJob]: ...


class InMemoryJobStore:
    def __init__(self):
        self._jobs: dict[str, TuningJob] = {}
        self._lock = threading.Lock()

    def create(self, job: TuningJob) -> None:
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"Job {job.job_id} already exists")
            self._jobs[job.job_id] = copy.deepcopy(job)

    def get(self, job_id: str) -> TuningJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def update(self, job_id: str, mutate_fn: Callable[[TuningJob], None]) -> TuningJob:
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")
            job = self._jobs[job_id]
            mutate_fn(job)
            return copy.deepcopy(job)

    def list_jobs(
        self, statuses: list[TuningJobStatus] | None = None, limit: int = 100
    ) -> list[TuningJob]:
        with self._lock:
            jobs = list(self._jobs.values())
            if statuses:
                allowed = {status.value for status in statuses}
                jobs = [j for j in jobs if j.status.value in allowed]
            # Sort by created_at descending
            sorted_jobs = sorted(jobs, key=lambda x: x.created_at, reverse=True)
            return [copy.deepcopy(j) for j in sorted_jobs[:limit]]

    def append_trial(self, job_id: str, trial: TrialRecord) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")
            self._jobs[job_id].trials.append(copy.deepcopy(trial))

    def list_trials(
        self,
        job_id: str,
        limit: int = 50,
        cursor: str | None = None,
        sort_by: str = "trial_number",
    ) -> tuple[list[TrialRecord], str | None]:
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")

            max_limit = max(1, min(limit, 200))
            trials = list(self._jobs[job_id].trials)
            if sort_by == "value":
                ranked = sorted(
                    trials,
                    key=lambda t: (t.value if t.value is not None else float("-inf")),
                    reverse=True,
                )
                return [copy.deepcopy(t) for t in ranked[:max_limit]], None

            ordered = sorted(trials, key=lambda t: t.trial_number)
            start = 0
            if cursor:
                try:
                    last_trial = int(cursor)
                except ValueError as exc:
                    raise ValueError(f"Invalid cursor: {cursor}") from exc
                for idx, trial in enumerate(ordered):
                    if trial.trial_number > last_trial:
                        start = idx
                        break
                else:
                    return [], None

            page = ordered[start : start + max_limit]
            if not page:
                return [], None
            next_cursor = (
                str(page[-1].trial_number) if start + max_limit < len(ordered) else None
            )
            return [copy.deepcopy(t) for t in page], next_cursor

    def set_heartbeat(self, job_id: str, ts: datetime) -> None:
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")
            self._jobs[job_id].heartbeat_at = ts
            self._jobs[job_id].updated_at = ts

    def list(self, limit: int = 50) -> list[TuningJob]:
        return self.list_jobs(limit=limit)
