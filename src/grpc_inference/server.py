"""Entrypoint for the grpc-inference service."""

from __future__ import annotations

import logging
import os
from concurrent import futures

import grpc

from api.model_manager import get_model_manager
from grpc_inference.config import GRPCInferenceConfig, load_config
from grpc_inference.logging import configure_logging
from grpc_inference.proto.inference.v1 import inference_pb2_grpc
from grpc_inference.service import InferenceService

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = load_config()
    _apply_env_overrides(config)

    manager = get_model_manager()
    if manager.load_production_model():
        logger.info(
            "loaded production model",
        )
    else:
        logger.warning("no production model loaded; using fallback scoring")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceService(config), server
    )

    listen_addr = f"{config.host}:{config.port}"
    server.add_insecure_port(listen_addr)
    logger.info("grpc-inference listening on %s", listen_addr)
    server.start()
    server.wait_for_termination()


def _apply_env_overrides(config: GRPCInferenceConfig) -> None:
    if config.mlflow_tracking_uri:
        os.environ.setdefault("MLFLOW_TRACKING_URI", config.mlflow_tracking_uri)


if __name__ == "__main__":
    main()
