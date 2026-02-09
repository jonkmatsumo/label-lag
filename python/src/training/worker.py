from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime, timedelta

from model.loader import DataLoader
from model.tuning import run_tuning_study
from training.job_queue import JobQueue
from training.job_store import JobStore
from training.jobs import TuningJobStatus

logger = logging.getLogger(__name__)


class TuningWorker:
    def __init__(self, job_store: JobStore, job_queue: JobQueue):
        self.job_store = job_store
        self.job_queue = job_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_job_id: str | None = None
        self._lock = threading.Lock()

    def start(self):
        """Start the background worker thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("TuningWorker started")

    def stop(self):
        """Stop the background worker thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("TuningWorker stopped")

    def _run(self):
        """Main loop of the worker thread."""
        while not self._stop_event.is_set():
            job_id = self.job_queue.get(block=True, timeout=1.0)
            if not job_id:
                continue

            with self._lock:
                self._current_job_id = job_id

            try:
                self._execute_job(job_id)
            except Exception as e:
                logger.exception(f"Job {job_id} failed with error: {e}")
                err_msg = str(e)

                def fail_job(j):
                    j.status = TuningJobStatus.FAILED
                    j.error_message = err_msg
                    j.ended_at = datetime.utcnow()

                self.job_store.update(job_id, fail_job)
            finally:
                with self._lock:
                    self._current_job_id = None
                self.job_queue.task_done()

    def _execute_job(self, job_id: str):
        """Execute a single tuning job."""
        job = self.job_store.get(job_id)
        if not job:
            return

        # Check if already canceled/canceling
        if job.status == TuningJobStatus.CANCELING:

            def cancel_job(j):
                j.status = TuningJobStatus.CANCELED
                j.ended_at = datetime.utcnow()

            self.job_store.update(job_id, cancel_job)
            return

        def start_job(j):
            j.status = TuningJobStatus.RUNNING
            j.started_at = datetime.utcnow()
            j.updated_at = datetime.utcnow()

        self.job_store.update(job_id, start_job)

        config = job.config
        training_window_days = config.get("training_window_days", 30)
        feature_columns = config.get("feature_columns")
        split_config = config.get("split_config")
        tuning_config = config.get("tuning_config")
        database_url = config.get("database_url")

        # Load data
        training_cutoff_date = datetime.now(UTC) - timedelta(days=training_window_days)
        loader = DataLoader(database_url=database_url)
        split = loader.load_train_test_split(
            training_cutoff_date,
            feature_columns=feature_columns,
            split_config=split_config,
        )

        if split.train_size == 0:
            raise ValueError("No training data available.")

        # Prepare tuning params
        v_frac = split_config.validation_fraction if split_config else 0.2
        val_size = max(5, int(split.train_size * v_frac))
        train_size = split.train_size - val_size

        if train_size < 10:
            raise ValueError(
                f"Insufficient training data for tuning: {train_size} rows"
            )

        x_tr = split.X_train.iloc[:train_size]
        y_tr = split.y_train.iloc[:train_size]
        x_val = split.X_train.iloc[train_size:]
        y_val = split.y_train.iloc[train_size:]

        n_negative = (y_tr == 0).sum()
        n_positive = (y_tr == 1).sum()
        scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

        import mlflow

        with mlflow.start_run(run_id=job.mlflow_run_id):
            best, trials_df = run_tuning_study(
                x_tr,
                y_tr,
                x_val,
                y_val,
                n_trials=tuning_config.n_trials,
                metric=tuning_config.metric,
                timeout_seconds=tuning_config.timeout_minutes * 60,
                seed=split_config.seed if split_config else 42,
                scale_pos_weight=scale_pos_weight,
                direction=tuning_config.direction,
                strategy=tuning_config.strategy,
                search_space_overrides=tuning_config.search_space,
                job_id=job_id,
                job_store=self.job_store,
            )

        # After completion, check status again (might have been canceled)
        final_job = self.job_store.get(job_id)
        if final_job.status == TuningJobStatus.CANCELING:

            def set_canceled(j):
                j.status = TuningJobStatus.CANCELED
                j.ended_at = datetime.utcnow()

            self.job_store.update(job_id, set_canceled)
        else:

            def set_completed(j):
                j.status = TuningJobStatus.COMPLETED
                j.ended_at = datetime.utcnow()

            self.job_store.update(job_id, set_completed)

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._current_job_id is not None
