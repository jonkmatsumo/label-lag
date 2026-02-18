"""Entrypoint for the grpc-inference service."""

from __future__ import annotations

import logging
import os
from concurrent import futures

import grpc

from inference.config import InferenceServerConfig, load_config
from inference.logging import configure_logging
from inference.proto.inference.v1 import (
    inference_pb2_grpc,
)
from inference.service import InferenceService

logger = logging.getLogger(__name__)


def serve() -> None:
    """Start gRPC server."""
    config: InferenceServerConfig = load_config()
    configure_logging()
    _apply_env_overrides(config)

    from forecast.model_manager import get_model_manager

    manager = get_model_manager()
    if manager.load_production_model():
        logger.info(
            "loaded production model",
        )
    else:
        logger.warning("no production model loaded; using fallback scoring")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceService(config), server
    )

    listen_addr = f"{config.host}:{config.port}"
    server.add_insecure_port(listen_addr)

    # Start Prometheus metrics server
    from prometheus_client import start_http_server

    start_http_server(config.metrics_port)
    logger.info(
        "inference metrics listening on %s (port %d)", config.host, config.metrics_port
    )

    logger.info("inference listening on %s", listen_addr)
    server.start()
    server.wait_for_termination()


def _apply_env_overrides(config: InferenceServerConfig) -> None:
    if config.mlflow_tracking_uri:
        os.environ.setdefault("MLFLOW_TRACKING_URI", config.mlflow_tracking_uri)


if __name__ == "__main__":
    serve()
