"""Rule store implementation using Analytics gRPC service."""

import json
import logging

from api.crud_client import get_crud_client
from api.proto.proto.crud.v1 import analytics_pb2
from rules_management.rules import Rule

logger = logging.getLogger(__name__)


class GrpcRuleStore:
    """Rule store that delegates storage to the Analytics gRPC service."""

    def __init__(self):
        """Initialize gRPC rule store."""
        pass

    def save(self, rule: Rule) -> None:
        """Save a rule via Analytics service."""
        client = get_crud_client()

        # Determine value_json
        if isinstance(rule.value, (list, dict)):
            value_json = json.dumps(rule.value)
        else:
            value_json = json.dumps(rule.value)

        rule_pb = analytics_pb2.Rule(
            id=rule.id,
            field=rule.field,
            op=rule.op,
            value_json=value_json,
            action=rule.action,
            score=rule.score if rule.score is not None else 0,
            severity=rule.severity,
            reason=rule.reason or "",
            status=rule.status,
        )

        client.stub.SaveRule(
            analytics_pb2.SaveRuleRequest(rule=rule_pb),
            timeout=client.timeout_seconds,
        )
        logger.debug(f"Saved rule {rule.id} via Analytics service")

    def get(self, rule_id: str) -> Rule | None:
        """Get a rule via Analytics service."""
        client = get_crud_client()
        try:
            resp = client.stub.GetRule(
                analytics_pb2.GetRuleRequest(rule_id=rule_id),
                timeout=client.timeout_seconds,
            )
            return self._from_pb(resp.rule)
        except Exception as e:
            logger.debug(f"Rule {rule_id} not found or error: {e}")
            return None

    def list_rules(
        self,
        status: str | None = None,
        include_archived: bool = False,
    ) -> list[Rule]:
        """List rules via Analytics service."""
        client = get_crud_client()
        req = analytics_pb2.ListRulesRequest(
            status=status or "", include_archived=include_archived
        )
        resp = client.stub.ListRules(
            req,
            timeout=client.timeout_seconds,
        )
        return [self._from_pb(r) for r in resp.rules]

    def delete(self, rule_id: str) -> bool:
        """Archive a rule via Analytics service."""
        client = get_crud_client()
        try:
            client.stub.DeleteRule(
                analytics_pb2.DeleteRuleRequest(rule_id=rule_id),
                timeout=client.timeout_seconds,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete rule {rule_id}: {e}")
            return False

    def exists(self, rule_id: str) -> bool:
        """Check if a rule exists."""
        return self.get(rule_id) is not None

    def _from_pb(self, r) -> Rule:
        """Convert proto Rule to local Rule object."""
        try:
            value = json.loads(r.value_json)
        except json.JSONDecodeError:
            value = r.value_json

        return Rule(
            id=r.id,
            field=r.field,
            op=r.op,
            value=value,
            action=r.action,
            score=r.score if r.score != 0 else None,
            severity=r.severity,
            reason=r.reason,
            status=r.status,
        )
