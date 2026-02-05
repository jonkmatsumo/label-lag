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
    summary="Generate synthetic data",
)
async def generate_data(request: GenerateDataRequest) -> GenerateDataResponse:
    """Generate synthetic transaction data via Analytics service."""
    try:
        from google.protobuf.timestamp_pb2 import Timestamp

        from api.proto.proto.crud.v1 import analytics_pb2
        from synthetic_pipeline.generator import DataGenerator

        generator = DataGenerator()
        result = generator.generate_dataset_with_sequences(
            num_users=request.num_users,
            fraud_rate=request.fraud_rate,
        )

        fraud_count = sum(1 for r in result.records if r.is_fraudulent)
        client = get_crud_client()

        if request.drop_existing:
            client.clear_all_data()

        proto_records = []
        for r in result.records:
            ts = Timestamp()
            ts.FromDatetime(r.transaction_timestamp)

            ec_ts = Timestamp()
            if r.identity_changes.email_changed_at:
                ec_ts.FromDatetime(r.identity_changes.email_changed_at)

            pc_ts = Timestamp()
            if r.identity_changes.phone_changed_at:
                pc_ts.FromDatetime(r.identity_changes.phone_changed_at)

            proto_records.append(
                analytics_pb2.GeneratedRecord(
                    record_id=r.record_id,
                    user_id=r.user_id,
                    full_name=r.full_name,
                    email=r.email,
                    phone=r.phone,
                    transaction_timestamp=ts,
                    is_off_hours_txn=r.is_off_hours_txn,
                    available_balance=float(r.account.available_balance),
                    balance_to_transaction_ratio=float(
                        r.account.balance_to_transaction_ratio
                    ),
                    avg_available_balance_30d=float(
                        r.behavior.avg_available_balance_30d
                    ),
                    balance_volatility_z_score=float(
                        r.behavior.balance_volatility_z_score
                    ),
                    bank_connections_count_24h=r.connection.bank_connections_count_24h,
                    bank_connections_count_7d=r.connection.bank_connections_count_7d,
                    bank_connections_avg_30d=float(
                        r.connection.bank_connections_avg_30d
                    ),
                    amount=float(r.transaction.amount),
                    amount_to_avg_ratio=float(r.transaction.amount_to_avg_ratio),
                    merchant_risk_score=r.transaction.merchant_risk_score,
                    is_returned=r.transaction.is_returned,
                    email_changed_at=ec_ts
                    if r.identity_changes.email_changed_at
                    else None,
                    phone_changed_at=pc_ts
                    if r.identity_changes.phone_changed_at
                    else None,
                    is_fraudulent=r.is_fraudulent,
                    fraud_type=r.fraud_type or "",
                )
            )

        proto_metadata = []
        for m in result.metadata:
            fc_ts = Timestamp()
            if m.fraud_confirmed_at:
                fc_ts.FromDatetime(m.fraud_confirmed_at)

            proto_metadata.append(
                analytics_pb2.EvaluationMetadata(
                    user_id=m.user_id,
                    record_id=m.record_id,
                    sequence_number=m.sequence_number,
                    fraud_confirmed_at=fc_ts if m.fraud_confirmed_at else None,
                    is_pre_fraud=m.is_pre_fraud,
                    days_to_fraud=m.days_to_fraud or 0,
                    is_train_eligible=m.is_train_eligible,
                )
            )

        client.store_generated_data(records=proto_records, metadata=proto_metadata)
        mat_resp = client.materialize_features()

        return GenerateDataResponse(
            success=True,
            total_records=len(result.records),
            fraud_records=fraud_count,
            features_materialized=mat_resp.total_processed,
        )

    except Exception as e:
        logger.exception("Data generation failed")
        return GenerateDataResponse(success=False, error=str(e))


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
