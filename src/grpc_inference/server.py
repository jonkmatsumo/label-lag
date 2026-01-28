"""Entrypoint for the grpc-inference service."""

from grpc_inference.config import load_config
from grpc_inference.logging import configure_logging


def main() -> None:
    configure_logging()
    _ = load_config()


if __name__ == "__main__":
    main()
