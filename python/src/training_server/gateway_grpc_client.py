import json
import logging
import os
import uuid
from typing import Any

import grpc
from google.protobuf import struct_pb2

try:
    from training_server.proto.crud.v1 import analytics_pb2
    from training_server.proto.gateway.v1 import gateway_pb2, gateway_pb2_grpc
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import generated gRPC modules: {e}")
    # Re-raise or handle gracefully? For now, let it crash if imports fail
    # as client is useless without them.
    raise

logger = logging.getLogger(__name__)


class GatewayGrpcClient:
    """gRPC Client for calling the Go Inference Gateway."""

    def __init__(self, target: str = None, timeout: float = 5.0):
        self.target = target or os.getenv(
            "INFERENCE_GATEWAY_GRPC_ADDR", "inference:50505"
        )
        self.timeout = timeout
        self.channel = grpc.insecure_channel(self.target)
        self.stub = gateway_pb2_grpc.GatewayServiceStub(self.channel)

    def evaluate_rules(
        self,
        features: dict[str, Any],
        base_score: int,
        ruleset: dict[str, Any] = None,
        request_id: str = None,
        shadow_mode: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate rules via gRPC. Matches GatewayDecisionClient.evaluate_rules signature.
        """
        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:12]}"

        # Convert features to Struct
        features_pb = struct_pb2.Struct()
        features_pb.update(features)

        # Convert ruleset if provided
        ruleset_pb = None
        if ruleset:
            ruleset_pb = self._dict_to_ruleset_proto(ruleset)

        req = gateway_pb2.EvaluateRulesRequest(
            features=features_pb,
            base_score=int(base_score),
            ruleset=ruleset_pb,
            shadow_mode=shadow_mode,
        )

        metadata = [("x-request-id", request_id)]

        try:
            resp = self.stub.EvaluateRules(req, timeout=self.timeout, metadata=metadata)
            return self._proto_to_dict(resp)
        except grpc.RpcError as e:
            self._handle_grpc_error(e)

    def diff_rules(
        self,
        features: dict[str, Any],
        base_score: int,
        ruleset_a: dict[str, Any] = None,
        ruleset_b: dict[str, Any] = None,
        request_id: str = None,
        shadow_mode: bool = False,
    ) -> dict[str, Any]:
        """
        Compare rulesets via gRPC. Matches GatewayDecisionClient.diff_rules signature.
        """
        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:12]}"

        features_pb = struct_pb2.Struct()
        features_pb.update(features)

        rs_a_pb = self._dict_to_ruleset_proto(ruleset_a) if ruleset_a else None
        rs_b_pb = self._dict_to_ruleset_proto(ruleset_b) if ruleset_b else None

        req = gateway_pb2.EvaluateRulesDiffRequest(
            features=features_pb,
            base_score=int(base_score),
            ruleset_a=rs_a_pb,
            ruleset_b=rs_b_pb,
            shadow_mode=shadow_mode,
        )

        metadata = [("x-request-id", request_id)]

        try:
            resp: gateway_pb2.EvaluateRulesDiffResponse = self.stub.EvaluateRulesDiff(
                req, timeout=self.timeout, metadata=metadata
            )

            # Convert response to dict structure matching HTTP response
            return {
                "a": self._proto_to_dict(resp.a),
                "b": self._proto_to_dict(resp.b),
                "diff": {
                    "severity": resp.diff.severity,
                    "score_delta": resp.diff.score_delta,
                    "matched_rules_added": list(resp.diff.matched_rules_added),
                    "matched_rules_removed": list(resp.diff.matched_rules_removed),
                },
                "total_evaluation_time_ms": resp.total_evaluation_time_ms,
            }
        except grpc.RpcError as e:
            self._handle_grpc_error(e)

    def _dict_to_ruleset_proto(self, ruleset_dict: dict) -> gateway_pb2.RuleSet:
        if not ruleset_dict or "rules" not in ruleset_dict:
            return gateway_pb2.RuleSet()

        rules_pb = []
        for r in ruleset_dict.get("rules", []):
            # Map dict rule to analytics.proto Rule
            val = r.get("value")
            val_json = ""
            if val is not None:
                val_json = json.dumps(val)

            # Score is optional/nullable in dict. In proto it's int32.
            score = 0
            if r.get("score") is not None:
                score = int(r["score"])

            # Map status.
            status = r.get("status", "active")

            rule_pb = analytics_pb2.Rule(
                id=r.get("id", ""),
                field=r.get("field", ""),
                op=r.get("op", ""),
                value_json=val_json,
                action=r.get("action", ""),
                score=score,
                severity=r.get("severity", ""),
                reason=r.get("reason", ""),
                status=status,
            )
            rules_pb.append(rule_pb)

        return gateway_pb2.RuleSet(rules=rules_pb)

    def _proto_to_dict(self, resp: gateway_pb2.EvaluateRulesResponse) -> dict[str, Any]:
        return {
            "final_score": resp.final_score,
            "baseline_score": resp.baseline_score,
            "shadow_score": resp.shadow_score,
            "matched_rules": list(resp.matched_rules),
            "explanations": [self._explanation_to_dict(e) for e in resp.explanations],
            "shadow_matched_rules": list(resp.shadow_matched_rules),
            "shadow_explanations": [
                self._explanation_to_dict(e) for e in resp.shadow_explanations
            ],
            "rejected": resp.rejected,
            "ruleset_version": resp.ruleset_version,
            "evaluation_time_ms": resp.evaluation_time_ms,
        }

    def _explanation_to_dict(self, exp) -> dict[str, Any]:
        return {
            "rule_id": exp.rule_id,
            "severity": exp.severity,
            "reason": exp.reason,
            "explanation": exp.explanation,
            "action": exp.action,
            "score": exp.score,
            "score_delta": exp.score_delta,
        }

    def _handle_grpc_error(self, e: grpc.RpcError):
        code = e.code()
        details = e.details() or "Unknown error"
        logger.error(f"Gateway gRPC failed ({code}): {details}")

        if code == grpc.StatusCode.INVALID_ARGUMENT:
            raise ValueError(f"Gateway evaluation failed (400): {details}")

        # Map other errors to RuntimeError to match HTTP client
        raise RuntimeError(f"Gateway evaluation failed ({code}): {details}")
