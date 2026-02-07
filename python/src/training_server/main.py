"""FastAPI application for fraud signal forecasting.

This API provides idempotent risk assessment for transactions.
It does not modify transaction state - it only provides an prediction.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from forecast.model_manager import get_model_manager
from forecast.routes import router as forecast_router
from forecast.services import get_forecaster
from training.routes import router as training_router
from training_server.schemas import HealthResponse

# Configure logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup."""
    # Startup: Load the production model
    logger.info("Starting up - loading production model...")
    manager = get_model_manager()
    success = manager.load_production_model()

    if success:
        logger.info(
            f"Model loaded successfully: version={manager.model_version}, "
            f"source={manager.model_source}"
        )
    else:
        logger.warning("No model loaded - API will use rule-based evaluation only")

    yield

    # Shutdown: cleanup if needed
    logger.info("Shutting down...")


app = FastAPI(
    title="Fraud Signal API",
    description="Risk signal evaluation for fraud detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Include domain routers
app.include_router(forecast_router)
app.include_router(training_router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns:
        HealthResponse with status and model information.
    """
    manager = get_model_manager()
    forecaster = get_forecaster()

    # Use model manager version if available, otherwise fall back to forecaster
    version = (
        manager.model_version if manager.model_loaded else forecaster.model_version
    )

    return HealthResponse(
        status="healthy",
        model_loaded=manager.model_loaded,
        version=version,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
