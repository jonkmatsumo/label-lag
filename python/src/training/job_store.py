from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from typing import Protocol

from training.jobs import TuningJob


class JobStore(Protocol):
    def create(self, job: TuningJob) -> None: ...

    def get(self, job_id: str) -> TuningJob | None: ...

    def update(
        self, job_id: str, mutate_fn: Callable[[TuningJob], None]
    ) -> TuningJob: ...

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

    def list(self, limit: int = 50) -> list[TuningJob]:
        with self._lock:
            # Sort by created_at descending
            sorted_jobs = sorted(
                self._jobs.values(), key=lambda x: x.created_at, reverse=True
            )
            return [copy.deepcopy(j) for j in sorted_jobs[:limit]]
