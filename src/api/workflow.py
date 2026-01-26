"""State machine for rule lifecycle transitions."""

import logging
import os
from dataclasses import asdict

from api.audit import get_audit_logger
from api.rules import Rule

logger = logging.getLogger(__name__)


# Define allowed transitions
ALLOWED_TRANSITIONS: dict[str, list[str]] = {
    "draft": ["pending_review"],
    "pending_review": ["approved", "draft"],
    "approved": ["active", "draft"],
    "active": ["shadow", "disabled"],
    "shadow": ["active", "disabled"],
    "disabled": ["active", "archived"],
    "archived": [],  # Terminal state
}


class TransitionError(Exception):
    """Raised when a state transition is invalid."""

    pass


class RuleStateMachine:
    """State machine for managing rule lifecycle transitions."""

    def __init__(self, require_approval: bool = False):
        """Initialize state machine.

        Args:
            require_approval: If True, certain transitions require approver role.
        """
        self.require_approval = require_approval
        self._audit_logger = get_audit_logger()

    def can_transition(self, from_status: str, to_status: str) -> bool:
        """Check if a transition is allowed.

        Args:
            from_status: Current status.
            to_status: Desired status.

        Returns:
            True if transition is allowed, False otherwise.
        """
        if from_status not in ALLOWED_TRANSITIONS:
            return False

        return to_status in ALLOWED_TRANSITIONS[from_status]

    def transition(
        self,
        rule: Rule,
        new_status: str,
        actor: str,
        reason: str = "",
        approver: str | None = None,
        previous_actor: str | None = None,
    ) -> Rule:
        """Transition a rule to a new status.

        Args:
            rule: Rule to transition.
            new_status: Target status.
            actor: Who is making the transition.
            reason: Optional reason for the transition.
            approver: Optional approver (required for some transitions).
            previous_actor: Optional actor from prior transition (e.g. submitter).
                When provided for approval transitions, prevents approver ==
                previous_actor (self-approval). Otherwise checks approver != actor.

        Returns:
            Updated rule with new status.

        Raises:
            TransitionError: If transition is not allowed or requirements not met.
        """
        current_status = rule.status

        # Check if transition is allowed
        if not self.can_transition(current_status, new_status):
            allowed = ALLOWED_TRANSITIONS.get(current_status, [])
            raise TransitionError(
                f"Cannot transition '{rule.id}' {current_status} -> {new_status}. "
                f"Allowed from '{current_status}': {allowed}"
            )

        # Check if approval is required
        if self.require_approval and self._requires_approval(
            current_status, new_status
        ):
            if not approver:
                raise TransitionError(
                    f"'{current_status}' -> '{new_status}' requires approval. "
                    "Provide an approver."
                )
            # Prevent self-approval: approver cannot be submitter (previous_actor)
            # or, if unknown, cannot be same as actor
            if previous_actor is not None and approver == previous_actor:
                raise TransitionError(
                    f"Self-approval not allowed. Actor '{approver}' cannot "
                    "approve their own submission."
                )
            if previous_actor is None and approver == actor:
                raise TransitionError(
                    f"Self-approval not allowed. Actor '{actor}' cannot approve "
                    "their own transition."
                )

        # Create updated rule
        rule_dict = asdict(rule)
        rule_dict["status"] = new_status
        updated_rule = Rule(**rule_dict)

        # Log the transition
        after_state = {"status": new_status}
        if approver:
            after_state["approver"] = approver

        self._audit_logger.log(
            rule_id=rule.id,
            action="state_change",
            actor=actor,
            before_state={"status": current_status},
            after_state=after_state,
            reason=reason or f"State transition: {current_status} -> {new_status}",
        )

        logger.info(
            "Rule '%s' transitioned '%s' -> '%s' by %s",
            rule.id,
            current_status,
            new_status,
            actor,
        )

        return updated_rule

    def _requires_approval(self, from_status: str, to_status: str) -> bool:
        """Check if a transition requires approval.

        Args:
            from_status: Current status.
            to_status: Target status.

        Returns:
            True if approval is required.
        """
        # pending_review -> approved requires approval
        if from_status == "pending_review" and to_status == "approved":
            return True

        # disabled -> active requires approval
        if from_status == "disabled" and to_status == "active":
            return True

        # shadow -> active requires approval (after validation)
        if from_status == "shadow" and to_status == "active":
            return True

        return False

    def get_allowed_transitions(self, current_status: str) -> list[str]:
        """Get list of allowed transitions from a status.

        Args:
            current_status: Current status.

        Returns:
            List of allowed target statuses.
        """
        return ALLOWED_TRANSITIONS.get(current_status, [])


def create_state_machine(require_approval: bool | None = None) -> RuleStateMachine:
    """Create a state machine instance.

    Args:
        require_approval: If True, certain transitions require approver role.
            If None, reads from REQUIRE_APPROVAL env var (defaults to True).

    Returns:
        RuleStateMachine instance.
    """
    if require_approval is None:
        # Default to True, allow override via env var
        require_approval = os.getenv("REQUIRE_APPROVAL", "true").lower() == "true"
    return RuleStateMachine(require_approval=require_approval)
