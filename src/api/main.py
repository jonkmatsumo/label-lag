"""FastAPI application for fraud signal evaluation.

This API provides idempotent risk assessment for transactions.
It does not modify transaction state - it only provides an evaluation.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import HealthResponse, SignalRequest, SignalResponse
from api.services import get_evaluator

app = FastAPI(
    title="Fraud Signal API",
    description="Risk signal evaluation for fraud detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns:
        HealthResponse with status and model information.
    """
    evaluator = get_evaluator()
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version=evaluator.model_version,
    )


@app.post(
    "/evaluate/signal",
    response_model=SignalResponse,
    tags=["Evaluation"],
    summary="Evaluate fraud signal",
    description="""
Evaluate the fraud risk signal for a transaction.

This endpoint is **idempotent** - it only provides an assessment without
modifying any transaction state. The same input will produce consistent
scoring (deterministic per user_id).

The response includes:
- **score**: Risk score from 1 (lowest) to 99 (highest)
- **risk_components**: Factors contributing to the score
- **model_version**: Version of the scoring model

Scores are calibrated to match real-world fraud score distributions where:
- Scores 1-20 are very common (low risk)
- Scores 80+ are rare (high risk)
""",
    responses={
        200: {
            "description": "Successful evaluation",
            "content": {
                "application/json": {
                    "example": {
                        "request_id": "req_123xyz",
                        "score": 85,
                        "risk_components": [
                            {
                                "key": "velocity",
                                "label": "high_transaction_velocity",
                            },
                            {"key": "history", "label": "insufficient_history"},
                        ],
                        "model_version": "v1.0.0",
                    }
                }
            },
        },
        422: {"description": "Validation error"},
    },
)
async def evaluate_signal(request: SignalRequest) -> SignalResponse:
    """Evaluate fraud signal for a transaction.

    Args:
        request: Signal evaluation request with user_id, amount, currency,
            and client_transaction_id.

    Returns:
        SignalResponse with risk score and contributing factors.
    """
    try:
        evaluator = get_evaluator()
        return evaluator.evaluate(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {e!s}",
        ) from e


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
