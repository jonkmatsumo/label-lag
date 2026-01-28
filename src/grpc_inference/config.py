"""Configuration for grpc-inference service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GRPCInferenceConfig:
    host: str = "0.0.0.0"
    port: int = 50052


def load_config() -> GRPCInferenceConfig:
    return GRPCInferenceConfig()
