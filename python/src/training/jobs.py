from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TuningJobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELING = "CANCELING"
    CANCELED = "CANCELED"

    def is_terminal(self) -> bool:
        return self in (
            TuningJobStatus.COMPLETED,
            TuningJobStatus.FAILED,
            TuningJobStatus.CANCELED,
        )


@dataclass
class TrialRecord:
    trial_number: int
    state: str
    value: float | None = None
    params: dict[str, str] = field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_ms: float | None = None


@dataclass
class TuningJob:
    job_id: str
    config: dict
    status: TuningJobStatus = TuningJobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: datetime | None = None
    mlflow_run_id: str | None = None
    requested_by: str | None = None
    total_trials: int = 0
    completed_trials: int = 0
    pruned_trials: int = 0
    best_value: float | None = None
    best_params: dict[str, str] = field(default_factory=dict)
    error_message: str | None = None
    trials: list[TrialRecord] = field(default_factory=list)

    @classmethod
    def create(
        cls, config: dict, total_trials: int, mlflow_run_id: str | None = None
    ) -> TuningJob:
        return cls(
            job_id=str(uuid.uuid4()),
            config=config,
            total_trials=total_trials,
            mlflow_run_id=mlflow_run_id,
        )
