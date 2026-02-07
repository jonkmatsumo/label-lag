import logging
import os

logger = logging.getLogger(__name__)

_client = None


def get_gateway_client():
    """Get the singleton GatewayGrpcClient instance."""
    global _client
    if _client is None:
        from training_server.gateway_grpc_client import GatewayGrpcClient

        transport = os.getenv("TRAINING_GATEWAY_TRANSPORT", "grpc").lower()
        if transport != "grpc":
            logger.warning(
                f"Transport '{transport}' requested but legacy HTTP support "
                "is removed. Defaulting to gRPC."
            )

        try:
            _client = GatewayGrpcClient()
            logger.info("Using gRPC transport for Inference Gateway")
        except Exception as e:
            logger.error(f"Failed to initialize gRPC client: {e}")
            raise RuntimeError(f"Failed to initialize gRPC client: {e}")

    return _client


def reset_gateway_client():
    """Reset the singleton client (useful for testing)."""
    global _client
    _client = None
