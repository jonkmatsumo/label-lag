from __future__ import annotations

import logging
import os
import threading
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError

from model.loader import DataLoader
from model.tuning import run_tuning_study
from training.job_queue import JobQueue
from training.job_store import JobStore
from training.jobs import TuningJobStatus
from training.metrics import observe_training_job_cancellation
from training.optuna_resume import get_optuna_storage_url
from training.schemas import SplitConfig, TuningConfig
from training.tuning_startup import get_tuning_job_retention_days, prune_tuning_jobs

logger = logging.getLogger(__name__)


def _get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_dict(cfg):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    return dict(cfg.__dict__)


def _coerce_model(cfg, model_cls):
    if cfg is None or not isinstance(cfg, dict):
        return cfg
    try:
        return model_cls(**cfg)
    except ValidationError:
        # Persisted or test payloads may intentionally bypass strict schema bounds.
        return model_cls.model_construct(**cfg)


class TuningWorker:
    def __init__(
        self,
        job_store: JobStore,
        job_queue: JobQueue,
        heartbeat_interval_seconds: int | None = None,
    ):
        self.job_store = job_store
        self.job_queue = job_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_job_id: str | None = None
        self._lock = threading.Lock()
        configured_interval = int(os.getenv("TUNING_HEARTBEAT_INTERVAL_SECONDS", "5"))
        interval = heartbeat_interval_seconds or configured_interval
        self._heartbeat_interval_seconds = max(1, interval)
        self._prune_interval_seconds = 15 * 60
        self._tuning_job_retention_days = get_tuning_job_retention_days()
        self._next_prune_at = datetime.now(UTC)

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
            self._maybe_prune_jobs()
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
                    j.ended_at = datetime.now(UTC)

                self.job_store.update(job_id, fail_job)
                self._log_lifecycle_event("failed", job_id)
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
            canceled_at = datetime.now(UTC)

            def cancel_job(j):
                j.status = TuningJobStatus.CANCELED
                j.ended_at = canceled_at

            self.job_store.update(job_id, cancel_job)
            canceled_job = self.job_store.get(job_id)
            if canceled_job:
                observe_training_job_cancellation(
                    canceled_job,
                    canceled_at=canceled_at,
                )
            self._log_lifecycle_event("canceled", job_id)
            return
        if job.status == TuningJobStatus.CANCELED:
            self._log_lifecycle_event("canceled", job_id)
            return
        if job.status.is_terminal():
            return

        def start_job(j):
            j.status = TuningJobStatus.RUNNING
            now = datetime.now(UTC)
            j.started_at = j.started_at or now
            j.heartbeat_at = now
            j.updated_at = now

        self.job_store.update(job_id, start_job)
        self._set_heartbeat(job_id)
        self._log_lifecycle_event("started", job_id)

        heartbeat_stop = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(job_id, heartbeat_stop),
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            config = job.config
            training_window_days = config.get("training_window_days", 30)
            feature_columns = config.get("feature_columns")
            split_cfg_obj = _coerce_model(config.get("split_config"), SplitConfig)
            tuning_cfg_obj = _coerce_model(config.get("tuning_config"), TuningConfig)
            database_url = config.get("database_url")
            optuna_storage_url = (
                config.get("optuna_storage_url") or get_optuna_storage_url()
            )

            if self._cancel_if_requested(job_id):
                return

            # Load data
            training_cutoff_date = datetime.now(UTC) - timedelta(
                days=training_window_days
            )
            loader = DataLoader(database_url=database_url)
            split = loader.load_train_test_split(
                training_cutoff_date,
                feature_columns=feature_columns,
                split_config=split_cfg_obj,
            )

            if split.train_size == 0:
                raise ValueError("No training data available.")
            max_dataset_rows = int(os.getenv("MAX_TUNING_DATASET_ROWS", "2000000"))
            if split.train_size > max_dataset_rows:
                raise ValueError(
                    "Training data exceeds MAX_TUNING_DATASET_ROWS "
                    f"({split.train_size} > {max_dataset_rows})"
                )

            # Prepare tuning params
            v_frac = _get(split_cfg_obj, "validation_fraction", 0.2)
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

            y_is_negative = y_tr == 0
            y_is_positive = y_tr == 1
            n_negative = y_is_negative.sum() if hasattr(y_is_negative, "sum") else 0
            n_positive = y_is_positive.sum() if hasattr(y_is_positive, "sum") else 0
            scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0

            if self._cancel_if_requested(job_id):
                return

            import mlflow

            with mlflow.start_run(run_id=job.mlflow_run_id):
                best, trials_df = run_tuning_study(
                    x_tr,
                    y_tr,
                    x_val,
                    y_val,
                    n_trials=_get(tuning_cfg_obj, "n_trials", 20),
                    metric=_get(tuning_cfg_obj, "metric", "pr_auc"),
                    timeout_seconds=_get(tuning_cfg_obj, "timeout_minutes", 30) * 60,
                    seed=_get(split_cfg_obj, "seed", 42),
                    scale_pos_weight=scale_pos_weight,
                    direction=_get(tuning_cfg_obj, "direction", "maximize"),
                    strategy=_get(tuning_cfg_obj, "strategy", "bayesian"),
                    search_space_overrides=_as_dict(
                        _get(tuning_cfg_obj, "search_space")
                    ),
                    job_id=job_id,
                    job_store=self.job_store,
                    storage_url=optuna_storage_url,
                    split_config=_as_dict(split_cfg_obj),
                    dataset_identity=config.get("dataset_identity"),
                )

            # After completion, check status again (might have been canceled)
            final_job = self.job_store.get(job_id)
            if final_job.status == TuningJobStatus.CANCELING:
                canceled_at = datetime.now(UTC)

                def set_canceled(j):
                    j.status = TuningJobStatus.CANCELED
                    j.ended_at = canceled_at

                self.job_store.update(job_id, set_canceled)
                canceled_job = self.job_store.get(job_id)
                if canceled_job:
                    observe_training_job_cancellation(
                        canceled_job,
                        canceled_at=canceled_at,
                    )
                self._log_lifecycle_event("canceled", job_id)
            else:

                def set_completed(j):
                    j.status = TuningJobStatus.COMPLETED
                    j.ended_at = datetime.now(UTC)

                self.job_store.update(job_id, set_completed)
                self._log_lifecycle_event("completed", job_id)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)
            self._set_heartbeat(job_id)

    def _heartbeat_loop(self, job_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(self._heartbeat_interval_seconds):
            self._set_heartbeat(job_id)

    def _maybe_prune_jobs(self) -> None:
        now = datetime.now(UTC)
        if now < self._next_prune_at:
            return
        self._next_prune_at = now + timedelta(seconds=self._prune_interval_seconds)
        try:
            removed = prune_tuning_jobs(
                self.job_store,
                retention_days=self._tuning_job_retention_days,
                now=now,
            )
            if removed:
                logger.info(
                    "tuning_job_pruned removed_jobs=%s retention_days=%s",
                    removed,
                    self._tuning_job_retention_days,
                )
        except Exception:
            logger.exception("Failed to prune terminal tuning jobs")

    def _set_heartbeat(self, job_id: str) -> None:
        try:
            self.job_store.set_heartbeat(job_id, datetime.now(UTC))
        except Exception:
            logger.exception("Failed to update heartbeat for job %s", job_id)

    def _log_lifecycle_event(self, event: str, job_id: str) -> None:
        job = self.job_store.get(job_id)
        if not job:
            return
        duration_seconds = None
        if job.started_at and job.ended_at:
            duration_seconds = (job.ended_at - job.started_at).total_seconds()
        elif job.started_at:
            duration_seconds = (datetime.now(UTC) - job.started_at).total_seconds()
        logger.info(
            "tuning_job_lifecycle event=%s job_id=%s status=%s duration_seconds=%.3f "
            "completed_trials=%s total_trials=%s pruned_trials=%s",
            event,
            job_id,
            job.status.value,
            duration_seconds or 0.0,
            job.completed_trials,
            job.total_trials,
            job.pruned_trials,
        )

    def _cancel_if_requested(self, job_id: str) -> bool:
        job = self.job_store.get(job_id)
        if not job:
            return True
        if job.status != TuningJobStatus.CANCELING:
            return False

        canceled_at = datetime.now(UTC)

        def set_canceled(current_job):
            current_job.status = TuningJobStatus.CANCELED
            current_job.updated_at = canceled_at
            current_job.ended_at = canceled_at

        self.job_store.update(job_id, set_canceled)
        canceled_job = self.job_store.get(job_id)
        if canceled_job:
            observe_training_job_cancellation(
                canceled_job,
                canceled_at=canceled_at,
            )
        self._log_lifecycle_event("canceled", job_id)
        return True

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._current_job_id is not None
