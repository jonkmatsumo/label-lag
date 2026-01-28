"""gRPC service implementation for inference."""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Any

import grpc
from google.protobuf import struct_pb2

from api.schemas import SignalRequest
from api.services import SignalEvaluator
from api.model_manager import get_model_manager
from synthetic_pipeline.db.session import DatabaseSession

from grpc_inference.config import GRPCInferenceConfig
from grpc_inference.proto.inference.v1 import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)


class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, config: GRPCInferenceConfig):
        self._config = config
        self._db_session = (
            DatabaseSession(database_url=config.db_dsn)
            if config.db_dsn
            else DatabaseSession()
        )
        self._evaluator = SignalEvaluator(db_session=self._db_session)
        self._manager = get_model_manager()

    def score(self, request, context):
        return self.Score(request, context)

    def Score(
        self, request: inference_pb2.ScoreRequest, context: grpc.ServicerContext
    ) -> inference_pb2.ScoreResponse:
        if not request.user_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user_id is required")
        if request.amount <= 0:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "amount must be greater than 0")
        if not request.client_transaction_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "client_transaction_id is required",
            )

        request_id = request.request_id or _generate_request_id()
        currency = request.currency or "USD"

        try:
            signal_request = SignalRequest(
                user_id=request.user_id,
                amount=Decimal(str(request.amount)),
                currency=currency,
                client_transaction_id=request.client_transaction_id,
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid request: {exc}")

        features = self._evaluator._fetch_features(signal_request)
        model_loaded = self._manager.model_loaded
        if model_loaded and features.has_history:
            raw_prob = self._evaluator._predict_with_model(self._manager, features)
            model_version = self._manager.model_version
        else:
            raw_prob = self._evaluator._calculate_probability(features)
            model_version = self._evaluator.model_version

        model_score = self._evaluator._calibrate_score(raw_prob)

        response = inference_pb2.ScoreResponse(
            request_id=request_id,
            model_score=float(model_score),
            model_version=model_version,
            model_loaded=model_loaded,
        )

        if self._config.include_features_used:
            response.features_used.CopyFrom(_features_to_struct(features))

        return response


def _features_to_struct(features: Any) -> struct_pb2.Struct:
    payload = {
        "velocity_24h": features.velocity_24h,
        "amount_to_avg_ratio_30d": features.amount_to_avg_ratio_30d,
        "balance_volatility_z_score": features.balance_volatility_z_score,
        "bank_connections_24h": features.bank_connections_24h,
        "merchant_risk_score": features.merchant_risk_score,
        "has_history": features.has_history,
        "transaction_amount": float(features.transaction_amount),
    }
    struct_msg = struct_pb2.Struct()
    struct_msg.update(payload)
    return struct_msg


def _generate_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"
