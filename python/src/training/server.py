"""Entrypoint for the training-server gRPC service."""

from __future__ import annotations

import logging
import os
from concurrent import futures
from datetime import UTC, datetime

import grpc

from forecast.model_manager import get_model_manager
from forecast.service import ForecastService
from forecast.v1 import forecast_pb2_grpc
from training.config import load_config  # noqa: E402
from training.job_queue import JobQueue  # noqa: E402
from training.service import TrainingService  # noqa: E402
from training.tuning_startup import (  # noqa: E402
    build_tuning_job_store,
    prune_tuning_jobs,
    reconcile_stale_jobs,
    reenqueue_pending_jobs,
)
from training.v1 import training_pb2_grpc  # noqa: E402
from training.worker import TuningWorker  # noqa: E402

logger = logging.getLogger(__name__)


def serve():
    # Startup: Load the production model
    logger.info("Starting up - loading production model...")
    manager = get_model_manager()
    success = manager.load_production_model()

    if success:
        logger.info(
            f"Model loaded successfully: version={manager.model_version}, "
            f"source={manager.model_source}"
        )
    else:
        logger.warning("No model loaded - API will use rule-based evaluation only")

    # Replay any failed training reports from local spool
    try:
        from training.crud_client import get_crud_client

        get_crud_client().replay_spooled_reports()
    except Exception as e:
        logger.warning(f"Failed to replay spooled reports: {e}")

    # Initialize Job infrastructure
    config = load_config()
    job_store = build_tuning_job_store(config)
    job_queue = JobQueue()
    heartbeat_interval_seconds = int(
        os.getenv("TUNING_HEARTBEAT_INTERVAL_SECONDS", "5")
    )
    pruned_jobs = prune_tuning_jobs(job_store=job_store, now=datetime.now(UTC))

    reconciled_jobs = reconcile_stale_jobs(
        job_store=job_store,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        now=datetime.now(UTC),
    )
    requeued_jobs = reenqueue_pending_jobs(job_store=job_store, job_queue=job_queue)
    logger.info(
        "Tuning startup reconciliation complete: pruned=%s stale_failed=%s "
        "pending_requeued=%s",
        pruned_jobs,
        reconciled_jobs,
        requeued_jobs,
    )

    worker = TuningWorker(job_store, job_queue)
    worker.start()

    # Start gRPC server in main process
    # Adjust max_workers as needed config not available globally, assume load_config
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))

    # Register both services
    training_pb2_grpc.add_TrainingServiceServicer_to_server(
        TrainingService(job_store, job_queue), server
    )
    forecast_pb2_grpc.add_ForecastServiceServicer_to_server(ForecastService(), server)

    listen_addr = f"{config.host}:{config.port}"
    server.add_insecure_port(listen_addr)
    logger.info("training (gRPC) listening on %s", listen_addr)

    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Stopping servers...")
    finally:
        worker.stop()


if __name__ == "__main__":
    serve()
