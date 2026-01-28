"""Configuration for grpc-inference service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GRPCInferenceConfig:
    host: str
    port: int
    db_dsn: str | None
    mlflow_tracking_uri: str | None
    model_name: str | None
    model_stage: str | None
    include_features_used: bool
    score_timeout_ms: int
    config_path: str | None


def load_config() -> GRPCInferenceConfig:
    config_path = os.getenv("GRPC_INFERENCE_CONFIG_PATH")
    if config_path:
        _load_env_file(config_path)

    return GRPCInferenceConfig(
        host=os.getenv("GRPC_INFERENCE_HOST", "0.0.0.0"),
        port=_get_int("GRPC_INFERENCE_PORT", 50052),
        db_dsn=os.getenv("GRPC_INFERENCE_DB_DSN"),
        mlflow_tracking_uri=os.getenv("GRPC_INFERENCE_MLFLOW_TRACKING_URI"),
        model_name=os.getenv("GRPC_INFERENCE_MODEL_NAME"),
        model_stage=os.getenv("GRPC_INFERENCE_MODEL_STAGE", "Production"),
        include_features_used=_get_bool(
            "GRPC_INFERENCE_INCLUDE_FEATURES_USED", False
        ),
        score_timeout_ms=_get_int("GRPC_INFERENCE_SCORE_TIMEOUT_MS", 1500),
        config_path=config_path,
    )


def _get_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
