from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import grpc
import optuna

from model.tuning import JobProgressCallback
from training.job_queue import JobQueue
from training.job_store import InMemoryJobStore
from training.jobs import (
    MAX_ERROR_MESSAGE_LENGTH,
    MAX_PARAM_KEY_LENGTH,
    MAX_PARAM_VALUE_LENGTH,
    MAX_PARAMS_ITEMS,
    TrialRecord,
    TuningJob,
    TuningJobStatus,
)
from training.service import TrainingService
from training.tuning_startup import prune_tuning_jobs
from training.v1 import training_pb2


class FakeRpcAbortError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str):
        super().__init__()
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


class FakeContext:
    def abort(self, code: grpc.StatusCode, details: str):
        raise FakeRpcAbortError(code=code, details=details)


def _create_job(
    store: InMemoryJobStore,
    *,
    status: TuningJobStatus = TuningJobStatus.PENDING,
    created_at: datetime | None = None,
) -> TuningJob:
    job = TuningJob.create(
        config={"tuning_config": {"metric": "pr_auc", "direction": "maximize"}},
        total_trials=5,
    )
    if created_at:
        job.created_at = created_at
        job.updated_at = created_at
    job.status = status
    if status.is_terminal():
        terminal_at = created_at or datetime.now(UTC)
        job.ended_at = terminal_at
        job.updated_at = terminal_at
    store.create(job)
    return job


def test_list_tuning_jobs_paginates_and_filters_statuses():
    store = InMemoryJobStore()
    queue = JobQueue()
    service = TrainingService(store, queue)
    now = datetime.now(UTC)

    oldest = _create_job(
        store, status=TuningJobStatus.PENDING, created_at=now - timedelta(minutes=3)
    )
    middle = _create_job(
        store, status=TuningJobStatus.COMPLETED, created_at=now - timedelta(minutes=2)
    )
    newest = _create_job(
        store, status=TuningJobStatus.PENDING, created_at=now - timedelta(minutes=1)
    )

    page_one = service.ListTuningJobs(
        training_pb2.ListTuningJobsRequest(limit=2),
        FakeContext(),
    )
    assert [job.job_id for job in page_one.jobs] == [newest.job_id, middle.job_id]
    assert page_one.next_cursor

    page_two = service.ListTuningJobs(
        training_pb2.ListTuningJobsRequest(limit=2, cursor=page_one.next_cursor),
        FakeContext(),
    )
    assert len(page_two.jobs) == 1
    assert page_two.jobs[0].job_id == oldest.job_id
    assert page_two.next_cursor == ""

    filtered = service.ListTuningJobs(
        training_pb2.ListTuningJobsRequest(statuses=[TuningJobStatus.PENDING.value]),
        FakeContext(),
    )
    assert filtered.jobs
    assert all(job.status == TuningJobStatus.PENDING.value for job in filtered.jobs)


def test_get_queue_depth_reflects_queue_entries():
    store = InMemoryJobStore()
    queue = JobQueue()
    service = TrainingService(store, queue)

    queue.enqueue("job-1")
    queue.enqueue("job-2")
    queue.cancel("job-1")

    response = service.GetQueueDepth(training_pb2.GetQueueDepthRequest(), FakeContext())
    assert response.depth == 1


def test_cancel_tuning_job_pending_is_immediate_and_idempotent():
    store = InMemoryJobStore()
    queue = JobQueue()
    service = TrainingService(store, queue)

    job = _create_job(store, status=TuningJobStatus.PENDING)
    queue.enqueue(job.job_id)

    with patch("training.service.observe_training_job_cancellation") as mock_observe:
        first = service.CancelTuningJob(
            training_pb2.CancelTuningJobRequest(job_id=job.job_id),
            FakeContext(),
        )
        second = service.CancelTuningJob(
            training_pb2.CancelTuningJobRequest(job_id=job.job_id),
            FakeContext(),
        )

    assert first.status == TuningJobStatus.CANCELED.value
    assert first.ended_at > 0
    assert second.status == TuningJobStatus.CANCELED.value
    assert queue.get(block=False) is None
    mock_observe.assert_called_once()


def test_cancel_tuning_job_running_becomes_canceling_idempotently():
    store = InMemoryJobStore()
    queue = JobQueue()
    service = TrainingService(store, queue)

    running = _create_job(store, status=TuningJobStatus.RUNNING)

    first = service.CancelTuningJob(
        training_pb2.CancelTuningJobRequest(job_id=running.job_id),
        FakeContext(),
    )
    second = service.CancelTuningJob(
        training_pb2.CancelTuningJobRequest(job_id=running.job_id),
        FakeContext(),
    )

    assert first.status == TuningJobStatus.CANCELING.value
    assert second.status == TuningJobStatus.CANCELING.value


def _build_trial(number: int) -> MagicMock:
    now = datetime.now(UTC)
    trial = MagicMock()
    trial.number = number
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.value = 0.91
    trial.params = {"max_depth": 6, "learning_rate": 0.1}
    trial.datetime_start = now
    trial.datetime_complete = now + timedelta(seconds=1)
    return trial


def _build_study(trial_number: int) -> MagicMock:
    study = MagicMock()
    study.best_trial = MagicMock(number=trial_number)
    study.best_value = 0.91
    study.best_params = {"max_depth": 6, "learning_rate": 0.1}
    return study


def test_mlflow_nested_trial_runs_enabled(monkeypatch):
    monkeypatch.setenv("TUNING_MLFLOW_NESTED_RUNS", "true")
    store = InMemoryJobStore()
    job = _create_job(store, status=TuningJobStatus.RUNNING)
    callback = JobProgressCallback(job.job_id, store)

    trial = _build_trial(3)
    study = _build_study(trial_number=3)

    with (
        patch("mlflow.start_run") as start_run,
        patch("mlflow.set_tag") as set_tag,
        patch("mlflow.log_param") as log_param,
        patch("mlflow.log_metric") as log_metric,
    ):
        start_run.return_value.__enter__.return_value = MagicMock()
        callback(study, trial)

    start_run.assert_called_with(run_name="trial_3", nested=True)
    set_tag.assert_any_call("job_id", job.job_id)
    log_param.assert_any_call("max_depth", 6)
    log_metric.assert_called_with("objective_value", 0.91)


def test_mlflow_nested_trial_runs_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TUNING_MLFLOW_NESTED_RUNS", raising=False)
    store = InMemoryJobStore()
    job = _create_job(store, status=TuningJobStatus.RUNNING)
    callback = JobProgressCallback(job.job_id, store)

    trial = _build_trial(4)
    study = _build_study(trial_number=4)

    with patch("mlflow.start_run") as start_run:
        callback(study, trial)

    start_run.assert_not_called()


def test_prune_tuning_jobs_removes_only_old_terminal_jobs():
    now = datetime.now(UTC)
    store = InMemoryJobStore()

    old_done = _create_job(
        store,
        status=TuningJobStatus.COMPLETED,
        created_at=now - timedelta(days=20),
    )
    _create_job(
        store,
        status=TuningJobStatus.RUNNING,
        created_at=now - timedelta(days=20),
    )
    _create_job(
        store,
        status=TuningJobStatus.FAILED,
        created_at=now - timedelta(days=1),
    )

    removed = prune_tuning_jobs(store, retention_days=14, now=now)

    assert removed == 1
    assert store.get(old_done.job_id) is None
    assert len(store.list_jobs(limit=10_000)) == 2


def test_trial_row_cap_applies_in_memory_store(monkeypatch):
    monkeypatch.setenv("TUNING_MAX_TRIAL_ROWS_PER_JOB", "2")
    store = InMemoryJobStore()
    job = _create_job(store)

    now = datetime.now(UTC)
    for trial_number in range(3):
        store.append_trial(
            job.job_id,
            TrialRecord(
                trial_number=trial_number,
                state="TrialState.COMPLETE",
                value=0.8,
                params={"max_depth": str(trial_number)},
                started_at=now,
                ended_at=now,
                duration_ms=25.0,
            ),
        )

    trials, _ = store.list_trials(job.job_id, limit=10)
    assert [trial.trial_number for trial in trials] == [1, 2]


def test_tuning_status_and_trial_payloads_are_bounded():
    store = InMemoryJobStore()
    queue = JobQueue()
    service = TrainingService(store, queue)

    job = _create_job(store)
    huge_error = "x" * (MAX_ERROR_MESSAGE_LENGTH + 500)
    huge_params = {
        f"key-{idx}-{'k' * 300}": "v" * 800 for idx in range(MAX_PARAMS_ITEMS + 20)
    }

    def set_large_fields(current_job):
        current_job.error_message = huge_error
        current_job.best_params = huge_params

    store.update(job.job_id, set_large_fields)
    store.append_trial(
        job.job_id,
        TrialRecord(
            trial_number=0,
            state="TrialState.COMPLETE",
            value=0.85,
            params=huge_params,
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
            duration_ms=100.0,
        ),
    )

    status = service.GetTuningStatus(
        training_pb2.GetTuningStatusRequest(job_id=job.job_id),
        FakeContext(),
    )
    trials = service.ListTrials(
        training_pb2.ListTrialsRequest(job_id=job.job_id),
        FakeContext(),
    )

    assert len(status.error_message) == MAX_ERROR_MESSAGE_LENGTH
    assert len(status.best_params) == MAX_PARAMS_ITEMS
    assert all(len(key) <= MAX_PARAM_KEY_LENGTH for key in status.best_params)
    assert all(
        len(value) <= MAX_PARAM_VALUE_LENGTH for value in status.best_params.values()
    )

    assert len(trials.trials[0].params) == MAX_PARAMS_ITEMS
    assert all(
        len(key) <= MAX_PARAM_KEY_LENGTH for key in trials.trials[0].params.keys()
    )
    assert all(
        len(value) <= MAX_PARAM_VALUE_LENGTH
        for value in trials.trials[0].params.values()
    )
