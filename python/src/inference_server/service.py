"""gRPC service implementation for forecasting."""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Any

import grpc
from google.protobuf import struct_pb2

from forecast_server.model_manager import get_model_manager
from forecast_server.services import SignalForecaster
from inference_server.config import InferenceServerConfig
from inference_server.proto.inference.v1 import (
    inference_pb2,
    inference_pb2_grpc,
)
from training_server.schemas import SignalRequest

logger = logging.getLogger(__name__)


class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    """Implementation of the gRPC InferenceService."""

    def __init__(self, config: InferenceServerConfig):
        self._config = config
        self._forecaster = SignalForecaster()
        self._manager = get_model_manager()

    def score(self, request, context):
        return self.Score(request, context)

    def Score(  # noqa: N802
        self, request: inference_pb2.ScoreRequest, context: grpc.ServicerContext
    ) -> inference_pb2.ScoreResponse:
        if not request.user_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "user_id is required")
        if request.amount <= 0:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "amount must be greater than 0"
            )
        if not request.client_transaction_id:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "client_transaction_id is required",
            )

        request.request_id or _generate_request_id()
        currency = request.currency or "USD"

        try:
            signal_request = SignalRequest(
                user_id=request.user_id,
                amount=Decimal(str(request.amount)),
                currency=currency,
                client_transaction_id=request.client_transaction_id,
                fallback_mode=request.fallback_mode
                if request.HasField("fallback_mode")
                else None,
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid request: {exc}")

        try:
            # Extract features from context if provided
            features_override = None
            if request.context and request.context.fields:
                from google.protobuf.json_format import MessageToDict

                features_override = MessageToDict(request.context)
                logger.debug(
                    f"Received context features: {list(features_override.keys())}"
                )

            prediction = self._forecaster.predict(
                signal_request, features_override=features_override
            )
        except Exception as exc:
            context.abort(grpc.StatusCode.INTERNAL, f"prediction failed: {exc}")

        response = inference_pb2.ScoreResponse(
            request_id=prediction["request_id"],
            model_score=float(prediction["model_score"]),
            model_version=prediction["model_version"],
            model_loaded=prediction["model_loaded"],
            fallback_used=prediction["fallback_used"],
        )

        if self._config.include_features_used:
            # We still need features for the features_used field
            features = self._forecaster._fetch_features(signal_request)
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
