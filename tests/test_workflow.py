"""Tests for rule state machine and transitions."""

import pytest

from api.audit import AuditLogger, set_audit_logger
from api.rules import Rule
from api.workflow import RuleStateMachine, TransitionError, create_state_machine


class TestStateMachineTransitions:
    """Tests for state transition logic."""

    @pytest.fixture
    def state_machine(self):
        """Create a state machine for testing."""
        # Use in-memory audit logger for tests
        test_logger = AuditLogger()
        set_audit_logger(test_logger)
        return RuleStateMachine(require_approval=False)

    @pytest.fixture
    def sample_rule(self):
        """Create a sample rule."""
        return Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="draft",
        )

    def test_can_transition_draft_to_pending_review(self, state_machine):
        """Test that draft can transition to pending_review."""
        assert state_machine.can_transition("draft", "pending_review") is True

    def test_can_transition_pending_review_to_approved(self, state_machine):
        """Test that pending_review can transition to approved."""
        assert state_machine.can_transition("pending_review", "approved") is True

    def test_can_transition_approved_to_active(self, state_machine):
        """Test that approved can transition to active."""
        assert state_machine.can_transition("approved", "active") is True

    def test_can_transition_approved_to_draft(self, state_machine):
        """Test that approved can transition back to draft."""
        assert state_machine.can_transition("approved", "draft") is True

    def test_cannot_transition_pending_review_to_active(self, state_machine):
        """Test that pending_review cannot directly transition to active."""
        assert state_machine.can_transition("pending_review", "active") is False

    def test_can_transition_pending_review_to_draft(self, state_machine):
        """Test that pending_review can transition back to draft."""
        assert state_machine.can_transition("pending_review", "draft") is True

    def test_can_transition_active_to_disabled(self, state_machine):
        """Test that active can transition to disabled."""
        assert state_machine.can_transition("active", "disabled") is True

    def test_can_transition_active_to_shadow(self, state_machine):
        """Test that active can transition to shadow."""
        assert state_machine.can_transition("active", "shadow") is True

    def test_can_transition_shadow_to_active(self, state_machine):
        """Test that shadow can transition to active."""
        assert state_machine.can_transition("shadow", "active") is True

    def test_can_transition_disabled_to_archived(self, state_machine):
        """Test that disabled can transition to archived."""
        assert state_machine.can_transition("disabled", "archived") is True

    def test_cannot_transition_archived(self, state_machine):
        """Test that archived cannot transition (terminal state)."""
        assert state_machine.can_transition("archived", "active") is False
        assert state_machine.can_transition("archived", "draft") is False

    def test_cannot_transition_draft_to_active(self, state_machine):
        """Test that draft cannot directly transition to active."""
        assert state_machine.can_transition("draft", "active") is False

    def test_cannot_transition_active_to_draft(self, state_machine):
        """Test that active cannot transition back to draft."""
        assert state_machine.can_transition("active", "draft") is False

    def test_get_allowed_transitions(self, state_machine):
        """Test getting list of allowed transitions."""
        transitions = state_machine.get_allowed_transitions("draft")
        assert "pending_review" in transitions
        assert len(transitions) == 1

        transitions = state_machine.get_allowed_transitions("active")
        assert "shadow" in transitions
        assert "disabled" in transitions
        assert len(transitions) == 2


class TestStateMachineTransitionExecution:
    """Tests for executing state transitions."""

    @pytest.fixture
    def state_machine(self):
        """Create a state machine for testing."""
        test_logger = AuditLogger()
        set_audit_logger(test_logger)
        return RuleStateMachine(require_approval=False)

    @pytest.fixture
    def draft_rule(self):
        """Create a draft rule."""
        return Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="draft",
        )

    def test_transition_draft_to_pending_review(self, state_machine, draft_rule):
        """Test transitioning draft to pending_review."""
        updated = state_machine.transition(
            draft_rule, "pending_review", actor="user123", reason="Ready for review"
        )

        assert updated.status == "pending_review"
        assert updated.id == draft_rule.id  # Other fields unchanged

    def test_transition_pending_review_to_approved(self, state_machine):
        """Test transitioning pending_review to approved."""
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="pending_review",
        )

        updated = state_machine.transition(
            rule, "approved", actor="approver1", reason="Approved"
        )

        assert updated.status == "approved"

    def test_transition_logs_audit_record(self, draft_rule):
        """Test that transitions are logged to audit trail."""
        test_logger = AuditLogger()
        set_audit_logger(test_logger)

        # Create state machine after setting logger
        state_machine = RuleStateMachine(require_approval=False)

        _ = state_machine.transition(
            draft_rule, "pending_review", actor="user123", reason="Ready for review"
        )

        audit_records = test_logger.get_rule_history("test_rule")
        assert len(audit_records) == 1
        assert audit_records[0].action == "state_change"
        assert audit_records[0].actor == "user123"
        assert audit_records[0].before_state["status"] == "draft"
        assert audit_records[0].after_state["status"] == "pending_review"

    def test_invalid_transition_raises_error(self, state_machine, draft_rule):
        """Test that invalid transitions raise TransitionError."""
        with pytest.raises(TransitionError, match="Cannot transition"):
            state_machine.transition(draft_rule, "active", actor="user123")

    def test_transition_preserves_rule_fields(self, state_machine, draft_rule):
        """Test that transition preserves all rule fields except status."""
        updated = state_machine.transition(
            draft_rule, "pending_review", actor="user123"
        )

        assert updated.id == draft_rule.id
        assert updated.field == draft_rule.field
        assert updated.op == draft_rule.op
        assert updated.value == draft_rule.value
        assert updated.action == draft_rule.action
        assert updated.score == draft_rule.score
        assert updated.status != draft_rule.status  # Only status changed


class TestStateMachineApproval:
    """Tests for approval requirements."""

    def test_requires_approval_pending_to_approved(self):
        """Test that pending_review -> approved requires approval."""
        state_machine = RuleStateMachine(require_approval=True)
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="pending_review",
        )

        # Should raise without approver
        with pytest.raises(TransitionError, match="requires approval"):
            state_machine.transition(rule, "approved", actor="user123")

        # Should succeed with approver
        updated = state_machine.transition(
            rule, "approved", actor="user123", approver="approver1"
        )
        assert updated.status == "approved"

    def test_requires_approval_disabled_to_active(self):
        """Test that disabled -> active requires approval."""
        state_machine = RuleStateMachine(require_approval=True)
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="disabled",
        )

        with pytest.raises(TransitionError, match="requires approval"):
            state_machine.transition(rule, "active", actor="user123")

        updated = state_machine.transition(
            rule, "active", actor="user123", approver="approver1"
        )
        assert updated.status == "active"

    def test_no_approval_required_when_disabled(self):
        """Test that approval requirement can be disabled."""
        state_machine = RuleStateMachine(require_approval=False)
        rule = Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status="pending_review",
        )

        # Should work without approver when require_approval=False
        updated = state_machine.transition(rule, "approved", actor="user123")
        assert updated.status == "approved"


class TestCreateStateMachine:
    """Tests for create_state_machine helper."""

    def test_create_state_machine(self):
        """Test creating a state machine."""
        sm = create_state_machine(require_approval=False)
        assert isinstance(sm, RuleStateMachine)
        assert sm.require_approval is False

    def test_create_state_machine_with_approval(self):
        """Test creating a state machine with approval required."""
        sm = create_state_machine(require_approval=True)
        assert sm.require_approval is True
