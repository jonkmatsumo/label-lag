from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
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


def _utc_now() -> datetime:
    return datetime.now(UTC)


MAX_ERROR_MESSAGE_LENGTH = 2000
MAX_PARAMS_ITEMS = 100
MAX_PARAM_KEY_LENGTH = 128
MAX_PARAM_VALUE_LENGTH = 512


def truncate_error_message(message: str | None) -> str | None:
    if message is None:
        return None
    return message[:MAX_ERROR_MESSAGE_LENGTH]


def bound_params(params: dict[str, str] | None) -> dict[str, str]:
    if not params:
        return {}

    bounded: dict[str, str] = {}
    for idx, (key, value) in enumerate(params.items()):
        if idx >= MAX_PARAMS_ITEMS:
            break
        bounded[str(key)[:MAX_PARAM_KEY_LENGTH]] = str(value)[:MAX_PARAM_VALUE_LENGTH]
    return bounded


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
    created_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    updated_at: datetime = field(default_factory=_utc_now)
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
