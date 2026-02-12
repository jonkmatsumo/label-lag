"""Entrypoint for the training-server gRPC service."""

from __future__ import annotations

import logging
import os
import sys
from concurrent import futures

import grpc

# Add generated proto directories to sys.path BEFORE importing the stubs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend(
    [
        os.path.join(BASE_DIR, "inference/proto"),
        os.path.join(BASE_DIR, "training/proto"),
        os.path.join(BASE_DIR, "forecast/proto"),
    ]
)

# Now it's safe to import the stubs
from forecast.proto.forecast.v1 import forecast_pb2_grpc  # noqa: E402

from forecast.model_manager import get_model_manager  # noqa: E402
from forecast.service import ForecastService  # noqa: E402
from training.config import load_config  # noqa: E402
from training.job_queue import JobQueue  # noqa: E402
from training.job_store import InMemoryJobStore  # noqa: E402
from training.proto.training.v1 import training_pb2_grpc  # noqa: E402
from training.service import TrainingService  # noqa: E402
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
    job_store = InMemoryJobStore()
    job_queue = JobQueue()
    worker = TuningWorker(job_store, job_queue)
    worker.start()

    # Start gRPC server in main process
    # Adjust max_workers as needed config not available globally, assume load_config
    config = load_config()
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
