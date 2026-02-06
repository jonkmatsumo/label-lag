"""Configuration for inference backend routing.

This module provides configuration for selecting between different
inference backends:

- "python" (default): Use Python FastAPI backend directly
- "go": Use Go inference gateway
- "go_with_fallback": Use Go gateway with Python fallback on failure

Usage:
    from api.inference_backend_config import get_inference_backend

    backend = get_inference_backend()  # Returns "python", "go", or "go_with_fallback"
"""

import os
from typing import Literal

InferenceBackend = Literal["python", "go", "go_with_fallback"]


def get_inference_backend() -> InferenceBackend:
    """Get the configured inference backend.

    Returns:
        The backend from INFERENCE_BACKEND env var, defaulting to "python".
    """
    backend = os.getenv("INFERENCE_BACKEND", "python")
    if backend not in ("python", "go", "go_with_fallback"):
        # Invalid backend, fall back to python
        return "python"
    return backend  # type: ignore[return-value]


def is_go_backend_enabled() -> bool:
    """Check if Go backend is enabled (either primary or with fallback).

    Returns:
        True if INFERENCE_BACKEND is "go" or "go_with_fallback".
    """
    return get_inference_backend() in ("go", "go_with_fallback")


def is_fallback_enabled() -> bool:
    """Check if Python fallback is enabled for Go backend failures.

    Returns:
        True if INFERENCE_BACKEND is "go_with_fallback".
    """
    return get_inference_backend() == "go_with_fallback"
