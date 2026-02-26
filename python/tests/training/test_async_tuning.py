from unittest.mock import MagicMock, patch

from training.job_queue import JobQueue
from training.job_store import InMemoryJobStore
from training.jobs import TuningJob, TuningJobStatus
from training.worker import TuningWorker


def test_job_store_basic():
    store = InMemoryJobStore()
    job = TuningJob.create(config={"test": 1}, total_trials=10)
    store.create(job)

    retrieved = store.get(job.job_id)
    assert retrieved.job_id == job.job_id
    assert retrieved.config == {"test": 1}
    assert retrieved.status == TuningJobStatus.PENDING

    def update_status(j):
        j.status = TuningJobStatus.RUNNING

    store.update(job.job_id, update_status)
    updated = store.get(job.job_id)
    assert updated.status == TuningJobStatus.RUNNING


def test_job_queue_basic():
    queue = JobQueue()
    queue.enqueue("job-1")
    assert queue.get(block=False) == "job-1"
    assert queue.get(block=False) is None


@patch("training.worker.DataLoader")
@patch("training.worker.run_tuning_study")
@patch("mlflow.start_run")
def test_worker_execution(mock_mlflow, mock_run_tuning, mock_loader):
    mock_run_tuning.return_value = ({}, MagicMock(attrs={}))

    mock_loader_instance = mock_loader.return_value
    mock_split = MagicMock()
    mock_split.train_size = 100
    mock_split.X_train = MagicMock()
    mock_split.y_train = MagicMock()
    # Mocking boolean indexing for y_tr == 0
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
    queue.enqueue(job.job_id)

    # Manually trigger one iteration of _execute_job
    worker._execute_job(job.job_id)

    updated_job = store.get(job.job_id)
    assert updated_job.status == TuningJobStatus.COMPLETED
    assert updated_job.started_at is not None
    assert updated_job.ended_at is not None


def test_job_cancellation_logic():
    store = InMemoryJobStore()
    queue = JobQueue()
    worker = TuningWorker(store, queue)

    job = TuningJob.create(config={}, total_trials=10)
    job.status = TuningJobStatus.CANCELING
    store.create(job)

    with patch("training.worker.observe_training_job_cancellation") as mock_observe:
        worker._execute_job(job.job_id)

    updated_job = store.get(job.job_id)
    assert updated_job.status == TuningJobStatus.CANCELED
    mock_observe.assert_called_once()


@patch("training.worker.DataLoader")
@patch("training.worker.run_tuning_study")
@patch("mlflow.start_run")
def test_worker_updates_heartbeat(mock_mlflow, mock_run_tuning, mock_loader):
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
    original_set_heartbeat = store.set_heartbeat
    store.set_heartbeat = MagicMock(side_effect=original_set_heartbeat)
    queue = JobQueue()
    worker = TuningWorker(store, queue, heartbeat_interval_seconds=1)

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

    updated_job = store.get(job.job_id)
    assert updated_job.heartbeat_at is not None
    assert store.set_heartbeat.called
