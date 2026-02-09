"""Helpers for standardized gRPC error responses."""

import grpc


def abort_invalid_argument(
    context: grpc.ServicerContext,
    field: str,
    message: str,
    help_text: str | None = None,
) -> None:
    """Abort request with structured INVALID_ARGUMENT error."""
    error_msg = f"Invalid argument for field '{field}': {message}"
    if help_text:
        error_msg += f" (Hint: {help_text})"

    context.abort(grpc.StatusCode.INVALID_ARGUMENT, error_msg)
