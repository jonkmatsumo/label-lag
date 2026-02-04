"""Inference event sink using Analytics gRPC service."""

import logging
from datetime import timezone

from google.protobuf.timestamp_pb2 import Timestamp

from api.crud_client import get_crud_client
from api.inference_log import InferenceEvent
from api.proto.proto.crud.v1 import analytics_pb2

logger = logging.getLogger(__name__)


class GrpcInferenceSink:
    """Inference event sink that delegates to the Analytics gRPC service."""

    def __init__(self):
        """Initialize gRPC inference sink."""
        pass

    def log_event(self, event: InferenceEvent) -> None:
        """Log an inference event via Analytics service."""
        try:
            client = get_crud_client()

            ts = Timestamp()
            ts.FromDatetime(
                event.timestamp
                if event.timestamp.tzinfo
                else event.timestamp.replace(tzinfo=timezone.utc)
            )

            impacts_pb = []
            for imp in event.rule_impacts:
                impacts_pb.append(
                    analytics_pb2.RuleImpact(
                        rule_id=imp.rule_id,
                        is_shadow=imp.is_shadow,
                        score_delta=imp.score_delta,
                    )
                )

            event_pb = analytics_pb2.InferenceEvent(
                request_id=event.request_id,
                timestamp=ts,
                model_version=event.model_version,
                rules_version=event.rules_version,
                model_score=event.model_score,
                final_score=event.final_score,
                rule_impacts=impacts_pb,
            )

            client.stub.LogInferenceEvent(
                analytics_pb2.LogInferenceEventRequest(event=event_pb),
                timeout=client.timeout_seconds,
            )
            logger.debug(
                f"Logged inference event {event.request_id} via Analytics service"
            )

        except Exception as e:
            logger.error(f"Failed to log inference event via Analytics service: {e}")
