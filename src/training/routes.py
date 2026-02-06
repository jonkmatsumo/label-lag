import logging

from fastapi import APIRouter

from api.crud_client import get_crud_client
from api.schemas import (
    ClearDataResponse,
    GenerateDataRequest,
    GenerateDataResponse,
    TrainRequest,
    TrainResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/data/generate",
    response_model=GenerateDataResponse,
    tags=["Data Management"],
    summary="Generate synthetic data [DEPRECATED]",
)
async def generate_data(request: GenerateDataRequest) -> GenerateDataResponse:
    """Deprecated: Use Go gateway /analytics/generate instead.

    This endpoint is kept for backwards compatibility but no longer performs
    data generation. The Go implementation in analytics-crud is now the
    sole data generation path.
    """
    return GenerateDataResponse(
        success=False,
        error="DEPRECATED: Use Go gateway /analytics/generate. "
              "This Python endpoint is no longer active.",
    )


@router.delete(
    "/data/clear",
    response_model=ClearDataResponse,
    tags=["Data Management"],
    summary="Clear all data",
)
async def clear_data() -> ClearDataResponse:
    """Clear all data from the database via Analytics service."""
    try:
        client = get_crud_client()
        resp = client.clear_all_data()
        return ClearDataResponse(
            success=resp.success,
            tables_cleared=list(resp.tables_cleared),
        )

    except Exception as e:
        logger.exception("Data clearing failed")
        return ClearDataResponse(success=False, error=str(e))


@router.post(
    "/train",
    response_model=TrainResponse,
    tags=["Training"],
    summary="Train a new model",
)
async def train_model_endpoint(request: TrainRequest) -> TrainResponse:
    """Train a new model with specified parameters."""
    try:
        from model.train import train_model

        run_id = train_model(
            max_depth=request.max_depth,
            training_window_days=request.training_window_days,
            feature_columns=request.selected_feature_columns,
            split_config=request.split_config,
            n_estimators=request.n_estimators,
            learning_rate=request.learning_rate,
            min_child_weight=request.min_child_weight,
            subsample=request.subsample,
            colsample_bytree=request.colsample_bytree,
            gamma=request.gamma,
            reg_alpha=request.reg_alpha,
            reg_lambda=request.reg_lambda,
            random_state=request.random_state,
            early_stopping_rounds=request.early_stopping_rounds,
            tuning_config=request.tuning_config,
        )
        return TrainResponse(success=True, run_id=run_id)
    except ValueError as e:
        return TrainResponse(success=False, error=str(e))
    except Exception as e:
        logger.exception("Training failed")
        return TrainResponse(success=False, error=str(e))
