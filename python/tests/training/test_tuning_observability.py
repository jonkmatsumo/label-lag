from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from training.job_queue import JobQueue
from training.job_store import InMemoryJobStore
from training.jobs import TuningJob
from training.service import TrainingService
from training.v1 import training_pb2
from training.worker import TuningWorker


class FailOnAbortContext:
    def abort(self, code, details):
        raise AssertionError(f"Unexpected abort: {code} {details}")


def _start_request() -> training_pb2.TrainRequest:
    return training_pb2.TrainRequest(
        training_window_days=30,
        selected_feature_columns=["f1"],
        feature_resolution_mode="strict",
        split_config=training_pb2.SplitConfig(
            strategy="temporal",
            validation_fraction=0.2,
            seed=42,
        ),
        tuning_config=training_pb2.TuningConfig(
            enabled=True,
            strategy="bayesian",
            n_trials=5,
            timeout_minutes=30,
            metric="pr_auc",
            direction="maximize",
        ),
    )


@patch("training.service.mlflow.set_experiment")
@patch("training.service.mlflow.start_run")
def test_start_tuning_logs_queue_depth(
    mock_start_run, _mock_set_experiment, monkeypatch, caplog
):
    monkeypatch.setattr("training.service.DataLoader.FEATURE_COLUMNS", ["f1"])
    caplog.set_level(logging.INFO)

    run = MagicMock()
    run.info.run_id = "run-123"
    mock_start_run.return_value.__enter__.return_value = run

    service = TrainingService(InMemoryJobStore(), JobQueue())
    response = service.StartTuningJob(_start_request(), FailOnAbortContext())

    assert response.job_id
    assert any(
        "tuning_job_enqueued" in record.message and "queue_depth=1" in record.message
        for record in caplog.records
    )


@patch("training.worker.DataLoader")
@patch("training.worker.run_tuning_study")
@patch("mlflow.start_run")
def test_worker_logs_lifecycle_events(
    mock_mlflow, mock_run_tuning, mock_loader, caplog
):
    caplog.set_level(logging.INFO)
    mock_run_tuning.return_value = ({}, MagicMock(attrs={}))

    mock_loader_instance = mock_loader.return_value
    mock_split = MagicMock()
    mock_split.train_size = 100
    mock_split.X_train = MagicMock()
    mock_split.y_train = MagicMock()
    mock_split.y_train.__eq__.return_value = MagicMock()
    mock_split.y_train.__eq__.return_value.sum.return_value = 50
    mock_loader_instance.load_train_test_split.return_value = mock_split

    store = InMemoryJobStore()
    queue = JobQueue()
    worker = TuningWorker(store, queue)

    job = TuningJob.create(
        config={
            "training_window_days": 30,
            "tuning_config": {
                "n_trials": 2,
                "metric": "pr_auc",
                "timeout_minutes": 1,
                "direction": "maximize",
                "strategy": "bayesian",
                "search_space": {},
            },
            "split_config": {"validation_fraction": 0.2, "seed": 42},
        },
        total_trials=2,
        mlflow_run_id="run-123",
    )
    store.create(job)

    worker._execute_job(job.job_id)

    assert any(
        "tuning_job_lifecycle event=started" in record.message
        for record in caplog.records
    )
    assert any(
        "tuning_job_lifecycle event=completed" in record.message
        and "duration_seconds=" in record.message
        for record in caplog.records
    )
