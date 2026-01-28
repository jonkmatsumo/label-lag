"""Logging setup for grpc-inference service."""

import logging


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
