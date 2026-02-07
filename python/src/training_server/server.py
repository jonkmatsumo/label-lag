"""Entrypoint for the training-server gRPC service."""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
from concurrent import futures

import grpc
import uvicorn

# Add generated proto directories to sys.path BEFORE importing the stubs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend(
    [
        os.path.join(BASE_DIR, "inference_server/proto"),
        os.path.join(BASE_DIR, "training_server/proto"),
        os.path.join(BASE_DIR, "forecast_server/proto"),
    ]
)

# Now it's safe to import the stubs
from forecast_server.model_manager import get_model_manager  # noqa: E402
from forecast_server.proto.forecast.v1 import forecast_pb2_grpc  # noqa: E402
from forecast_server.service import ForecastService  # noqa: E402
from training_server.config import TrainingServerConfig, load_config  # noqa: E402
from training_server.proto.training.v1 import training_pb2_grpc  # noqa: E402
from training_server.service import TrainingService  # noqa: E402

logger = logging.getLogger(__name__)


def start_http() -> None:
    """Start FastAPI HTTP server."""
    config: TrainingServerConfig = load_config()
    # We use training_server.main:app to avoid circular imports
    uvicorn.run(
        "training_server.main:app", host=config.host, port=8000, log_level="info"
    )


def serve() -> None:
    """Start both gRPC and HTTP servers."""
    config: TrainingServerConfig = load_config()
    logging.basicConfig(level=logging.INFO)

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

    # Start HTTP server in a separate process
    http_process = multiprocessing.Process(target=start_http)
    http_process.daemon = True
    http_process.start()

    # Start gRPC server in main process
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))

    # Register both services
    training_pb2_grpc.add_TrainingServiceServicer_to_server(TrainingService(), server)
    forecast_pb2_grpc.add_ForecastServiceServicer_to_server(ForecastService(), server)

    listen_addr = f"{config.host}:{config.port}"
    server.add_insecure_port(listen_addr)
    logger.info("training_server (gRPC) listening on %s", listen_addr)

    try:
        server.start()
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Stopping servers...")
    finally:
        http_process.terminate()
        http_process.join()


if __name__ == "__main__":
    serve()
