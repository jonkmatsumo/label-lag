from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import grpc
import optuna
import pytest

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
from training.tuning_startup import reconcile_stale_jobs
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


def _service() -> tuple[TrainingService, InMemoryJobStore, JobQueue]:
    store = InMemoryJobStore()
    queue = JobQueue()
    return TrainingService(store, queue), store, queue


def _create_job(
    store: InMemoryJobStore,
    *,
    status: TuningJobStatus = TuningJobStatus.PENDING,
    total_trials: int = 5,
) -> TuningJob:
    job = TuningJob.create(
        config={
            "training_window_days": 30,
            "feature_columns": ["f1", "f2"],
            "split_config": {
                "strategy": "temporal",
                "validation_fraction": 0.2,
                "seed": 42,
            },
            "tuning_config": {
                "enabled": True,
                "direction": "maximize",
                "metric": "pr_auc",
            },
            "optuna_storage_url": "postgresql://localhost/test",
        },
        total_trials=total_trials,
        mlflow_run_id="run-tuning-parent",
    )
    job.status = status
    store.create(job)
    return job


def _trial_record(trial_number: int = 0) -> TrialRecord:
    now = datetime.now(UTC)
    return TrialRecord(
        trial_number=trial_number,
        state="TrialState.COMPLETE",
        value=0.91,
        params={"max_depth": "6", "learning_rate": "0.1"},
        started_at=now,
        ended_at=now + timedelta(seconds=1),
        duration_ms=1000.0,
    )


def test_resume_reconcile_moves_stale_running_to_pending(monkeypatch):
    now = datetime.now(UTC)
    store = InMemoryJobStore()
    job = _create_job(store, status=TuningJobStatus.RUNNING, total_trials=3)
    stale_heartbeat = now - timedelta(seconds=60)

    def mark_stale(current):
        current.heartbeat_at = stale_heartbeat
        current.updated_at = stale_heartbeat

    store.update(job.job_id, mark_stale)

    trial = MagicMock()
    trial.number = 0
    trial.state = optuna.trial.TrialState.COMPLETE
    trial.value = 0.91
    trial.params = {"max_depth": 6}
    trial.datetime_start = now - timedelta(seconds=20)
    trial.datetime_complete = now - timedelta(seconds=10)

    study = MagicMock()
    study.trials = [trial]
    study.best_trial = MagicMock(number=0)
    study.best_value = 0.91
    study.best_params = {"max_depth": 6}

    monkeypatch.setattr(
        "training.tuning_startup.load_existing_tuning_study",
        lambda **_: study,
    )
    monkeypatch.setattr(
        "training.tuning_startup.study_can_resume",
        lambda *_args, **_kwargs: True,
    )

    stale_failed = reconcile_stale_jobs(
        store,
        heartbeat_interval_seconds=5,
        now=now,
    )

    updated = store.get(job.job_id)
    assert stale_failed == 0
    assert updated.status == TuningJobStatus.PENDING
    assert updated.completed_trials == 1
    assert updated.best_value == 0.91
    assert updated.best_params.get("max_depth") == "6"


def test_promote_trial_rejects_unknown_job():
    service, _, _ = _service()

    with pytest.raises(FakeRpcAbortError) as exc:
        service.PromoteTrial(
            training_pb2.PromoteTrialRequest(job_id="missing", trial_number=1),
            FakeContext(),
        )

    assert exc.value.code() == grpc.StatusCode.NOT_FOUND


def test_promote_trial_rejects_missing_trial():
    service, store, _ = _service()
    job = _create_job(store, status=TuningJobStatus.COMPLETED)

    with pytest.raises(FakeRpcAbortError) as exc:
        service.PromoteTrial(
            training_pb2.PromoteTrialRequest(job_id=job.job_id, trial_number=999),
            FakeContext(),
        )

    assert exc.value.code() == grpc.StatusCode.NOT_FOUND


def test_promote_trial_dry_run_returns_plan():
    service, store, _ = _service()
    job = _create_job(store, status=TuningJobStatus.COMPLETED)
    store.append_trial(job.job_id, _trial_record(2))

    response = service.PromoteTrial(
        training_pb2.PromoteTrialRequest(
            job_id=job.job_id,
            trial_number=2,
            model_name="fraud-model-staging",
            dry_run=True,
        ),
        FakeContext(),
    )

    assert response.status == "COMPLETED"
    assert "dry-run" in response.error_message
    assert "fraud-model-staging" in response.error_message


@patch("training.service.mlflow.MlflowClient")
@patch("training.service.train_model")
def test_promote_trial_triggers_retrain_with_trial_overrides(
    mock_train_model,
    mock_mlflow_client_cls,
):
    mock_train_model.return_value = "run-promoted-123"
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [
        MagicMock(run_id="run-promoted-123", version="7")
    ]
    mock_mlflow_client_cls.return_value = mock_client

    service, store, _ = _service()
    job = _create_job(store, status=TuningJobStatus.COMPLETED)
    store.append_trial(
        job.job_id,
        TrialRecord(
            trial_number=1,
            state="TrialState.COMPLETE",
            value=0.94,
            params={
                "max_depth": "8",
                "n_estimators": "120",
                "learning_rate": "0.05",
                "gamma": "0.25",
            },
        ),
    )

    response = service.PromoteTrial(
        training_pb2.PromoteTrialRequest(
            job_id=job.job_id,
            trial_number=1,
        ),
        FakeContext(),
    )

    assert response.status == "COMPLETED"
    assert response.mlflow_run_id == "run-promoted-123"
    assert response.model_version == "7"
    kwargs = mock_train_model.call_args.kwargs
    assert kwargs["max_depth"] == 8
    assert kwargs["n_estimators"] == 120
    assert kwargs["learning_rate"] == 0.05
    assert kwargs["gamma"] == 0.25


def test_get_tuning_job_info_returns_bounded_payload():
    service, store, _ = _service()
    job = _create_job(store, status=TuningJobStatus.COMPLETED)
    oversized_params = {
        f"key-{i}-{'k' * 400}": "v" * 900 for i in range(MAX_PARAMS_ITEMS + 15)
    }

    def set_big_fields(current):
        current.error_message = "x" * (MAX_ERROR_MESSAGE_LENGTH + 100)
        current.best_params = oversized_params

    store.update(job.job_id, set_big_fields)
    store.append_trial(
        job.job_id,
        TrialRecord(
            trial_number=0,
            state="TrialState.COMPLETE",
            value=0.83,
            params=oversized_params,
        ),
    )

    response = service.GetTuningJobInfo(
        training_pb2.GetTuningJobInfoRequest(job_id=job.job_id, trials_limit=50),
        FakeContext(),
    )

    assert len(response.job.error_message) == MAX_ERROR_MESSAGE_LENGTH
    assert len(response.trials) == 1
    assert len(response.trials[0].params) == MAX_PARAMS_ITEMS
    assert all(len(key) <= MAX_PARAM_KEY_LENGTH for key in response.trials[0].params)
    assert all(
        len(value) <= MAX_PARAM_VALUE_LENGTH
        for value in response.trials[0].params.values()
    )
    assert response.mlflow_links["tuning_run_id"] == "run-tuning-parent"


def test_admin_rpcs_denied_when_flag_disabled(monkeypatch):
    monkeypatch.delenv("ENABLE_TUNING_ADMIN_RPC", raising=False)
    service, _, _ = _service()

    with pytest.raises(FakeRpcAbortError) as exc:
        service.RequeueTuningJob(
            training_pb2.RequeueTuningJobRequest(job_id="x"),
            FakeContext(),
        )

    assert exc.value.code() == grpc.StatusCode.PERMISSION_DENIED


def test_requeue_admin_rpc_allowed_and_transitions_job(monkeypatch):
    monkeypatch.setenv("ENABLE_TUNING_ADMIN_RPC", "true")
    service, store, queue = _service()
    job = _create_job(store, status=TuningJobStatus.FAILED)

    with patch("training.service.load_existing_tuning_study", return_value=MagicMock()):
        response = service.RequeueTuningJob(
            training_pb2.RequeueTuningJobRequest(job_id=job.job_id),
            FakeContext(),
        )

    assert response.status == TuningJobStatus.PENDING.value
    assert store.get(job.job_id).status == TuningJobStatus.PENDING
    assert queue.depth() == 1


def test_finalize_admin_rpc_applies_terminal_status(monkeypatch):
    monkeypatch.setenv("ENABLE_TUNING_ADMIN_RPC", "true")
    service, store, _ = _service()
    job = _create_job(store, status=TuningJobStatus.RUNNING)

    response = service.FinalizeTuningJob(
        training_pb2.FinalizeTuningJobRequest(
            job_id=job.job_id,
            final_status=TuningJobStatus.FAILED.value,
            reason="operator finalized due to infrastructure maintenance",
        ),
        FakeContext(),
    )

    assert response.status == TuningJobStatus.FAILED.value
    updated = store.get(job.job_id)
    assert updated.status == TuningJobStatus.FAILED
    assert updated.ended_at is not None
    assert "operator finalized" in (updated.error_message or "")
