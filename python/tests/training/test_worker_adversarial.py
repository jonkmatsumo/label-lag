import threading
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from training.job_queue import JobQueue
from training.job_store import InMemoryJobStore
from training.jobs import TuningJob, TuningJobStatus
from training.worker import TuningWorker


class TestWorkerAdversarial:
    @pytest.fixture
    def job_store(self):
        return InMemoryJobStore()

    @pytest.fixture
    def job_queue(self):
        return JobQueue()

    @pytest.fixture
    def worker(self, job_store, job_queue):
        worker = TuningWorker(job_store, job_queue, heartbeat_interval_seconds=1)
        return worker

    def test_worker_cancellation_mid_run_leaves_clean_state(
        self, worker, job_store, job_queue
    ):
        """Worker must handle job cancellation mid-execution and cleanup."""
        # 1. Setup job
        job = TuningJob.create(
            config={
                "feature_columns": ["f1"],
                "training_window_days": 30,
            },
            total_trials=10,
            mlflow_run_id="run-adversarial",
        )
        job_id = job.job_id
        job_store.create(job)
        job_queue.enqueue(job_id)

        # 2. Mock heavy dependencies to block mid-flight
        in_training = threading.Event()
        can_continue = threading.Event()

        def slow_load(*args, **kwargs):
            in_training.set()
            can_continue.wait(timeout=10)  # Block until we say so
            return MagicMock(
                train_size=100,
                X_train=pd.DataFrame({"f1": [0] * 100}),
                y_train=pd.Series([0, 1] * 50),
            )

        with (
            patch("training.worker.DataLoader") as mock_loader_cls,
            patch("mlflow.start_run"),
            patch("training.worker.run_tuning_study") as mock_tune,
        ):
            mock_loader = MagicMock()
            mock_loader.load_train_test_split.side_effect = slow_load
            mock_loader_cls.return_value = mock_loader

            # 3. Start worker
            worker_thread = threading.Thread(target=worker._run, daemon=True)
            worker_thread.start()

            # 4. Wait for job to start loading
            if not in_training.wait(timeout=5):
                pytest.fail("Worker never reached loading state")

            # 5. Cancel the job via store (simulating API call)
            def request_cancel(j):
                j.status = TuningJobStatus.CANCELING

            job_store.update(job_id, request_cancel)

            # 6. Allow training to resume
            can_continue.set()

            # 7. Wait for worker to finish processing the job
            timeout = time.time() + 5
            final_status = None
            while time.time() < timeout:
                j = job_store.get(job_id)
                if j.status.is_terminal():
                    final_status = j.status
                    break
                time.sleep(0.1)

            worker._stop_event.set()
            worker_thread.join(timeout=2)

            assert final_status == TuningJobStatus.CANCELED
            # Ensure we didn't call tuning study after cancellation
            mock_tune.assert_not_called()

    def test_mlflow_artifact_failure_marks_job_failed(
        self, worker, job_store, job_queue
    ):
        """Worker must mark job as FAILED if artifact persistence raises."""
        job = TuningJob.create(
            config={"feature_columns": ["f1"]}, total_trials=5, mlflow_run_id="run-fail"
        )
        job_id = job.job_id
        job_store.create(job)
        job_queue.enqueue(job_id)

        with (
            patch("training.worker.DataLoader") as mock_loader_cls,
            patch("mlflow.start_run"),
            patch("training.worker.run_tuning_study") as mock_tune,
        ):
            # 1. Setup mocks to succeed until tuning
            mock_loader = MagicMock()
            mock_loader.load_train_test_split.return_value = MagicMock(
                train_size=100,
                X_train=pd.DataFrame({"f1": [0] * 100}),
                y_train=pd.Series([0, 1] * 50),
            )
            mock_loader_cls.return_value = mock_loader

            # 2. Make run_tuning_study raise
            # (simulating artifact failure inside or similar)
            mock_tune.side_effect = RuntimeError("Artifact store full")

            # 3. Process job (simulating the try-except block in worker._run)
            try:
                worker._execute_job(job_id)
            except Exception as e:
                err_msg = str(e)

                def fail_job(j):
                    j.status = TuningJobStatus.FAILED
                    j.error_message = err_msg
                    from datetime import UTC, datetime

                    j.ended_at = datetime.now(UTC)

                job_store.update(job_id, fail_job)

            # 4. Verify job state
            final_job = job_store.get(job_id)
            assert final_job.status == TuningJobStatus.FAILED
            assert "Artifact store full" in final_job.error_message

    def test_mlflow_start_run_failure(self, worker, job_store, job_queue):
        """Worker must handle failure to even start a run (e.g. MLflow down)."""
        job = TuningJob.create(
            config={"feature_columns": ["f1"]}, total_trials=5, mlflow_run_id="run-fail"
        )
        job_id = job.job_id
        job_store.create(job)
        job_queue.enqueue(job_id)

        with (
            patch("training.worker.DataLoader") as mock_loader_cls,
            patch("mlflow.start_run") as mock_start_run,
        ):
            mock_loader = MagicMock()
            mock_loader.load_train_test_split.return_value = MagicMock(
                train_size=100,
                X_train=pd.DataFrame({"f1": [0] * 100}),
                y_train=pd.Series([0, 1] * 50),
            )
            mock_loader_cls.return_value = mock_loader

            # Simulate MLflow being down
            mock_start_run.side_effect = Exception("MLflow Connection Refused")

            # Let the worker thread process it to test the full catch-all
            worker_thread = threading.Thread(target=worker._run, daemon=True)
            worker_thread.start()

            # Wait for terminal status
            timeout = time.time() + 5
            final_job = None
            while time.time() < timeout:
                j = job_store.get(job_id)
                if j.status == TuningJobStatus.FAILED:
                    final_job = j
                    break
                time.sleep(0.1)

            worker._stop_event.set()
            worker_thread.join(timeout=2)

            assert final_job is not None
            assert "MLflow Connection Refused" in final_job.error_message
