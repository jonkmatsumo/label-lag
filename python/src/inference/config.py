"""Configuration for pygrpc service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceServerConfig:
    host: str
    port: int
    metrics_port: int
    max_workers: int
    db_dsn: str | None
    mlflow_tracking_uri: str | None
    model_name: str | None
    model_stage: str | None
    include_features_used: bool
    score_timeout_ms: int
    config_path: str | None


def load_config() -> InferenceServerConfig:
    config_path = os.getenv("INFERENCE_SERVER_CONFIG_PATH")
    if config_path:
        _load_env_file(config_path)

    return InferenceServerConfig(
        host=os.getenv("INFERENCE_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("INFERENCE_SERVER_PORT", "50052")),
        metrics_port=int(os.getenv("INFERENCE_SERVER_METRICS_PORT", "9091")),
        max_workers=int(os.getenv("INFERENCE_SERVER_MAX_WORKERS", "10")),
        db_dsn=os.getenv("INFERENCE_SERVER_DB_DSN"),
        mlflow_tracking_uri=os.getenv("INFERENCE_SERVER_MLFLOW_TRACKING_URI"),
        model_name=os.getenv("INFERENCE_SERVER_MODEL_NAME"),
        model_stage=os.getenv("INFERENCE_SERVER_MODEL_STAGE", "Production"),
        include_features_used=_get_bool(
            "INFERENCE_SERVER_INCLUDE_FEATURES_USED", False
        ),
        score_timeout_ms=_get_int("INFERENCE_SERVER_SCORE_TIMEOUT_MS", 1500),
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
