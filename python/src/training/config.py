"""Configuration for training service."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingServerConfig:
    host: str
    port: int
    max_workers: int
    db_dsn: str | None
    mlflow_tracking_uri: str | None


def load_config() -> TrainingServerConfig:
    return TrainingServerConfig(
        host=os.getenv("TRAINING_SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("TRAINING_SERVER_PORT", "50053")),
        max_workers=int(os.getenv("TRAINING_SERVER_MAX_WORKERS", "10")),
        db_dsn=os.getenv("DATABASE_URL"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    )
