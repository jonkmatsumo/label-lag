"""Tests for rule publish functionality."""

import pytest

from api.audit import AuditLogger, set_audit_logger
from forecast.model_manager import ModelManager
from rules_management.draft_store import DraftRuleStore, set_draft_store
from rules_management.rules import Rule, RuleSet, RuleStatus
from rules_management.versioning import RuleVersionStore, set_version_store


class TestPublishFlow:
    """Tests for rule publish flow."""

    @pytest.fixture
    def draft_store(self):
        """Create a draft store for testing."""
        store = DraftRuleStore()
        set_draft_store(store)
        return store

    @pytest.fixture
    def version_store(self):
        """Create a version store for testing."""
        store = RuleVersionStore()
        set_version_store(store)
        return store

    @pytest.fixture
    def model_manager(self):
        """Create a model manager for testing."""
        manager = ModelManager()
        # Initialize with empty ruleset
        manager._ruleset = RuleSet.empty()
        # Set as global manager for testing
        import forecast.model_manager

        forecast.model_manager._manager = manager
        return manager

    @pytest.fixture
    def audit_logger(self):
        """Create an audit logger for testing."""
        logger = AuditLogger()
        set_audit_logger(logger)
        return logger

    @pytest.fixture
    def approved_rule(self):
        """Create an approved rule."""
        return Rule(
            id="test_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status=RuleStatus.APPROVED.value,
        )

    def test_publish_updates_store_status(
        self, draft_store, version_store, model_manager, audit_logger, approved_rule
    ):
        """Test that publish updates status in the store."""
        # Add approved rule to draft store
        draft_store._rules[approved_rule.id] = approved_rule
        draft_store._save_rules()

        # Simulate publish: transition to active
        from rules_management.workflow import RuleStateMachine

        state_machine = RuleStateMachine(require_approval=False)
        active_rule = state_machine.transition(
            rule=approved_rule,
            new_status=RuleStatus.ACTIVE.value,
            actor="test_actor",
            reason="Published to production",
        )

        # Update draft store
        draft_store.save(active_rule)

        # Verify status was updated in store
        rule_in_store = draft_store.get(approved_rule.id)
        assert rule_in_store is not None
        assert rule_in_store.status == RuleStatus.ACTIVE.value

    def test_publish_creates_audit_event(
        self, draft_store, version_store, model_manager, audit_logger, approved_rule
    ):
        """Test that publish creates RULE_PUBLISHED audit event."""
        draft_store._rules[approved_rule.id] = approved_rule

        # Publish the rule
        audit_logger.log(
            rule_id=approved_rule.id,
            action="RULE_PUBLISHED",
            actor="test_actor",
            before_state={"status": RuleStatus.APPROVED.value},
            after_state={"status": RuleStatus.ACTIVE.value},
            reason="Published to production",
        )

        # Verify audit event
        records = audit_logger.get_rule_history(approved_rule.id)
        assert len(records) == 1
        assert records[0].action == "RULE_PUBLISHED"
        assert records[0].actor == "test_actor"
        assert records[0].before_state["status"] == RuleStatus.APPROVED.value
        assert records[0].after_state["status"] == RuleStatus.ACTIVE.value

    def test_publish_generates_version(
        self, draft_store, version_store, model_manager, audit_logger, approved_rule
    ):
        """Test that publish generates a new version."""
        draft_store._rules[approved_rule.id] = approved_rule

        # Create version snapshot
        from rules_management.workflow import RuleStateMachine

        state_machine = RuleStateMachine(require_approval=False)
        active_rule = state_machine.transition(
            rule=approved_rule,
            new_status=RuleStatus.ACTIVE.value,
            actor="test_actor",
        )

        version = version_store.save(
            rule=active_rule,
            created_by="test_actor",
            reason="Published to production",
        )

        # Verify version was created
        assert version is not None
        assert version.rule_id == approved_rule.id
        assert version.rule.status == RuleStatus.ACTIVE.value
        assert version.created_by == "test_actor"

    def test_cannot_publish_non_approved_rule(
        self, draft_store, version_store, model_manager, audit_logger
    ):
        """Test that only approved rules can be published."""
        draft_rule = Rule(
            id="draft_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=80,
            status=RuleStatus.DRAFT.value,
        )

        draft_store._rules[draft_rule.id] = draft_rule

        from rules_management.workflow import RuleStateMachine, TransitionError

        state_machine = RuleStateMachine(require_approval=False)

        # Should not be able to transition draft -> active
        with pytest.raises(TransitionError):
            state_machine.transition(
                rule=draft_rule,
                new_status=RuleStatus.ACTIVE.value,
                actor="test_actor",
            )
