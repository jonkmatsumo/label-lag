"""Shared HTTP error helpers."""

from __future__ import annotations

import grpc
from fastapi import HTTPException


def analytics_http_exception(exc: Exception) -> HTTPException:
    """Map Analytics gRPC errors to HTTP exceptions."""
    if isinstance(exc, grpc.RpcError):
        code = exc.code()
        if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
            detail = exc.details() or "Analytics service unavailable"
            return HTTPException(status_code=503, detail=detail)

    return HTTPException(status_code=500, detail=str(exc))
