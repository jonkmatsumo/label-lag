"""Tests for rule lifecycle endpoints (approve, reject, activate, disable, shadow)."""

import os

import pytest
from fastapi.testclient import TestClient

from api.audit import AuditLogger, get_audit_logger, set_audit_logger
from api.draft_store import DraftRuleStore, set_draft_store
from api.main import app
from api.versioning import (
    RuleVersionStore,
    get_version_store,
    set_version_store,
)

# Set REQUIRE_APPROVAL=true for tests (default behavior)
os.environ["REQUIRE_APPROVAL"] = "true"


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset all stores before each test."""
    set_draft_store(DraftRuleStore())
    set_audit_logger(AuditLogger())
    set_version_store(RuleVersionStore())
    yield
    set_draft_store(DraftRuleStore())
    set_audit_logger(AuditLogger())
    set_version_store(RuleVersionStore())


def _create_and_submit_rule(client, rule_id: str, actor: str = "test_user") -> None:
    """Helper to create a draft rule and submit it for review."""
    # Create rule
    client.post(
        "/rules/draft",
        json={
            "id": rule_id,
            "field": "velocity_24h",
            "op": ">",
            "value": 5,
            "action": "clamp_min",
            "score": 70,
            "severity": "medium",
            "reason": "Test rule",
            "actor": actor,
        },
    )

    # Submit for review
    client.post(
        f"/rules/draft/{rule_id}/submit",
        json={
            "actor": actor,
            "justification": "Ready for review and approval",
        },
    )


class TestApproveDraftRule:
    """Tests for POST /rules/draft/{rule_id}/approve endpoint."""

    def test_approve_draft_rule_returns_200(self, client):
        """Test that approving a draft rule returns 200."""
        _create_and_submit_rule(client, "approve_test_001", actor="submitter_user")

        response = client.post(
            "/rules/draft/approve_test_001/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Rule looks good, approved for activation",
            },
        )
        assert response.status_code == 200

    def test_approve_draft_rule_response_structure(self, client):
        """Test that response has correct structure."""
        _create_and_submit_rule(client, "approve_test_002", actor="submitter_user")

        response = client.post(
            "/rules/draft/approve_test_002/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved for production use",
            },
        )
        data = response.json()

        assert "rule" in data
        assert "approved_at" in data

        rule = data["rule"]
        assert rule["rule_id"] == "approve_test_002"
        assert rule["status"] == "approved"

    def test_approve_changes_status_to_active(self, client):
        """Test that approval changes status to active."""
        _create_and_submit_rule(client, "approve_test_003", actor="submitter_user")

        # Verify it's pending_review
        get_response = client.get("/rules/draft/approve_test_003")
        assert get_response.json()["status"] == "pending_review"

        # Approve it
        client.post(
            "/rules/draft/approve_test_003/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        # Verify status changed to approved (not active - requires publish)
        get_response = client.get("/rules/draft/approve_test_003")
        assert get_response.json()["status"] == "approved"

    def test_approve_creates_version(self, client):
        """Test that approval creates a version snapshot."""
        _create_and_submit_rule(client, "approve_test_004", actor="submitter_user")

        client.post(
            "/rules/draft/approve_test_004/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        version_store = get_version_store()
        versions = version_store.list_versions("approve_test_004")
        assert len(versions) >= 2  # At least create + approve

        # Check latest version has active status
        latest = versions[-1]
        assert latest.rule.status == "active"

    def test_approve_creates_audit_record(self, client):
        """Test that approval creates an audit record."""
        _create_and_submit_rule(client, "approve_test_005", actor="submitter_user")

        client.post(
            "/rules/draft/approve_test_005/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved for activation",
            },
        )

        audit_logger = get_audit_logger()
        records = audit_logger.get_rule_history("approve_test_005")

        # Find approval state_change record
        approval_records = [
            r
            for r in records
            if r.action == "state_change" and r.after_state.get("status") == "active"
        ]
        assert len(approval_records) > 0

        approval_record = approval_records[-1]
        assert approval_record.actor == "approver_user"
        assert "approver" in approval_record.after_state
        assert approval_record.after_state["approver"] == "approver_user"

    def test_approve_requires_pending_review_status(self, client):
        """Test that only pending_review rules can be approved."""
        # Create but don't submit
        client.post(
            "/rules/draft",
            json={
                "id": "approve_test_006",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Test",
                "actor": "test_user",
            },
        )

        # Try to approve draft (should fail)
        response = client.post(
            "/rules/draft/approve_test_006/approve",
            json={
                "approver": "approver_user",
                "reason": "Should not work",
            },
        )
        assert response.status_code == 400
        assert "not pending review" in response.json()["detail"].lower()

    def test_approve_not_found_returns_404(self, client):
        """Test that approving non-existent rule returns 404."""
        response = client.post(
            "/rules/draft/nonexistent/approve",
            json={
                "approver": "approver_user",
                "reason": "Test",
            },
        )
        assert response.status_code == 404

    def test_approve_prevents_self_approval(self, client):
        """Test that actor cannot approve their own rule."""
        _create_and_submit_rule(client, "approve_test_007", actor="submitter_user")

        # Try to approve with same user as submitter (should fail)
        response = client.post(
            "/rules/draft/approve_test_007/approve",
            json={
                "approver": "submitter_user",  # Same as actor
                "reason": "Self approval attempt",
            },
        )
        assert response.status_code == 400
        assert "self-approval" in response.json()["detail"].lower()

        # Approve with different user (should work)
        response = client.post(
            "/rules/draft/approve_test_007/approve",
            json={
                "approver": "different_approver",
                "reason": "Approved by different user",
            },
        )
        assert response.status_code == 200

    def test_approve_requires_approver_when_enabled(self, client):
        """Test that approver is required when REQUIRE_APPROVAL=true."""
        _create_and_submit_rule(client, "approve_test_008", actor="submitter_user")

        # Try without approver (should fail)
        # Note: This tests the state machine, but the endpoint might handle it
        # differently. The state machine should raise TransitionError.
        response = client.post(
            "/rules/draft/approve_test_008/approve",
            json={
                "reason": "No approver provided",
            },
        )
        # Should fail validation or state machine check
        assert response.status_code in [400, 422]


class TestRejectDraftRule:
    """Tests for POST /rules/draft/{rule_id}/reject endpoint."""

    def test_reject_draft_rule_returns_200(self, client):
        """Test that rejecting a draft rule returns 200."""
        _create_and_submit_rule(client, "reject_test_001")

        response = client.post(
            "/rules/draft/reject_test_001/reject",
            json={
                "actor": "reviewer_user",
                "reason": "Rule needs more work before approval",
            },
        )
        assert response.status_code == 200

    def test_reject_changes_status_to_draft(self, client):
        """Test that rejection changes status back to draft."""
        _create_and_submit_rule(client, "reject_test_002")

        # Verify it's pending_review
        get_response = client.get("/rules/draft/reject_test_002")
        assert get_response.json()["status"] == "pending_review"

        # Reject it
        client.post(
            "/rules/draft/reject_test_002/reject",
            json={
                "actor": "reviewer_user",
                "reason": "Needs revision before approval",
            },
        )

        # Verify status changed back to draft
        get_response = client.get("/rules/draft/reject_test_002")
        assert get_response.json()["status"] == "draft"

    def test_reject_requires_reason(self, client):
        """Test that rejection requires reason (min 10 chars)."""
        _create_and_submit_rule(client, "reject_test_003")

        # Try with short reason (should fail)
        response = client.post(
            "/rules/draft/reject_test_003/reject",
            json={
                "actor": "reviewer_user",
                "reason": "short",  # Too short
            },
        )
        assert response.status_code == 422  # Validation error

        # Try with valid reason (should work)
        response = client.post(
            "/rules/draft/reject_test_003/reject",
            json={
                "actor": "reviewer_user",
                "reason": "This is a valid rejection reason with enough characters",
            },
        )
        assert response.status_code == 200

    def test_reject_creates_version(self, client):
        """Test that rejection creates a version snapshot."""
        _create_and_submit_rule(client, "reject_test_004")

        client.post(
            "/rules/draft/reject_test_004/reject",
            json={
                "actor": "reviewer_user",
                "reason": "Rejected for revision",
            },
        )

        version_store = get_version_store()
        versions = version_store.list_versions("reject_test_004")
        assert len(versions) >= 2  # At least create + reject

        # Check latest version has draft status
        latest = versions[-1]
        assert latest.rule.status == "draft"

    def test_reject_creates_audit_record(self, client):
        """Test that rejection creates an audit record."""
        _create_and_submit_rule(client, "reject_test_005")

        client.post(
            "/rules/draft/reject_test_005/reject",
            json={
                "actor": "reviewer_user",
                "reason": "Rejected: needs more validation",
            },
        )

        audit_logger = get_audit_logger()
        records = audit_logger.get_rule_history("reject_test_005")

        # Find rejection state_change record
        reject_records = [
            r
            for r in records
            if r.action == "state_change"
            and r.after_state.get("status") == "draft"
            and r.before_state.get("status") == "pending_review"
        ]
        assert len(reject_records) > 0

        reject_record = reject_records[-1]
        assert reject_record.actor == "reviewer_user"
        assert "Rejected" in reject_record.reason

    def test_reject_requires_pending_review_status(self, client):
        """Test that only pending_review rules can be rejected."""
        # Create but don't submit
        client.post(
            "/rules/draft",
            json={
                "id": "reject_test_006",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Test",
                "actor": "test_user",
            },
        )

        # Try to reject draft (should fail)
        response = client.post(
            "/rules/draft/reject_test_006/reject",
            json={
                "actor": "reviewer_user",
                "reason": "Should not work on draft",
            },
        )
        assert response.status_code == 400
        assert "not pending review" in response.json()["detail"].lower()


class TestActivateRule:
    """Tests for POST /rules/{rule_id}/activate endpoint."""

    def test_activate_from_pending_review(self, client):
        """Test that can activate from pending_review."""
        _create_and_submit_rule(client, "activate_test_001", actor="submitter_user")

        response = client.post(
            "/rules/activate_test_001/activate",
            json={
                "actor": "admin_user",
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Activating approved rule for production",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rule"]["status"] == "active"

    def test_activate_from_shadow(self, client):
        """Test that can activate from shadow (with approval)."""
        # Create, submit, approve to get to active
        _create_and_submit_rule(client, "activate_test_002", actor="submitter_user")
        client.post(
            "/rules/draft/activate_test_002/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        # Move to shadow
        client.post(
            "/rules/activate_test_002/shadow",
            json={
                "actor": "admin_user",
                "reason": "Moving to shadow for monitoring",
            },
        )

        # Activate from shadow
        response = client.post(
            "/rules/activate_test_002/activate",
            json={
                "actor": "admin_user",
                "approver": "approver_user",
                "reason": "Promoting from shadow to active",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rule"]["status"] == "active"

    def test_activate_from_disabled(self, client):
        """Test that can activate from disabled (with approval)."""
        # Create, submit, approve to get to active
        _create_and_submit_rule(client, "activate_test_003", actor="submitter_user")
        client.post(
            "/rules/draft/activate_test_003/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        # Disable it
        client.post(
            "/rules/activate_test_003/disable",
            json={
                "actor": "admin_user",
                "reason": "Temporarily disabled",
            },
        )

        # Activate from disabled
        response = client.post(
            "/rules/activate_test_003/activate",
            json={
                "actor": "admin_user",
                "approver": "approver_user",
                "reason": "Re-enabling after issue resolved",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rule"]["status"] == "active"

    def test_activate_requires_reason(self, client):
        """Test that activation requires reason (min 10 chars)."""
        _create_and_submit_rule(client, "activate_test_004", actor="submitter_user")

        # Try with short reason (should fail)
        response = client.post(
            "/rules/activate_test_004/activate",
            json={
                "actor": "admin_user",
                "approver": "approver_user",  # Different from submitter_user
                "reason": "short",  # Too short
            },
        )
        assert response.status_code == 422  # Validation error

        # Try with valid reason (should work)
        response = client.post(
            "/rules/activate_test_004/activate",
            json={
                "actor": "admin_user",
                "approver": "approver_user",  # Different from submitter_user
                "reason": "This is a valid activation reason",
            },
        )
        assert response.status_code == 200

    def test_activate_creates_version(self, client):
        """Test that activation creates a version snapshot."""
        _create_and_submit_rule(client, "activate_test_005", actor="submitter_user")

        client.post(
            "/rules/activate_test_005/activate",
            json={
                "actor": "admin_user",
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Activating rule",
            },
        )

        version_store = get_version_store()
        versions = version_store.list_versions("activate_test_005")
        assert len(versions) >= 2  # At least create + activate

        # Check latest version has active status
        latest = versions[-1]
        assert latest.rule.status == "active"

    def test_activate_creates_audit_record(self, client):
        """Test that activation creates an audit record."""
        _create_and_submit_rule(client, "activate_test_006", actor="submitter_user")

        client.post(
            "/rules/activate_test_006/activate",
            json={
                "actor": "admin_user",
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Activating for production use",
            },
        )

        audit_logger = get_audit_logger()
        records = audit_logger.get_rule_history("activate_test_006")

        # Find activation state_change record
        activate_records = [
            r
            for r in records
            if r.action == "state_change" and r.after_state.get("status") == "active"
        ]
        assert len(activate_records) > 0

        activate_record = activate_records[-1]
        assert activate_record.actor == "admin_user"
        assert "approver" in activate_record.after_state
        assert activate_record.after_state["approver"] == "approver_user"

    def test_activate_not_found_returns_404(self, client):
        """Test that activating non-existent rule returns 404."""
        response = client.post(
            "/rules/nonexistent/activate",
            json={
                "actor": "admin_user",
                "approver": "approver_user",
                "reason": "Test activation",
            },
        )
        assert response.status_code == 404

    def test_activate_invalid_status_returns_400(self, client):
        """Test that activating from invalid status returns 400."""
        # Create but don't submit (stays in draft)
        client.post(
            "/rules/draft",
            json={
                "id": "activate_test_007",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Test",
                "actor": "test_user",
            },
        )

        # Try to activate from draft (should fail)
        response = client.post(
            "/rules/activate_test_007/activate",
            json={
                "actor": "admin_user",
                "approver": "approver_user",
                "reason": "Cannot activate from draft status",
            },
        )
        assert response.status_code == 400
        assert "cannot activate" in response.json()["detail"].lower()


class TestDisableRule:
    """Tests for POST /rules/{rule_id}/disable endpoint."""

    def test_disable_from_active(self, client):
        """Test that can disable from active."""
        _create_and_submit_rule(client, "disable_test_001", actor="submitter_user")
        client.post(
            "/rules/draft/disable_test_001/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        response = client.post(
            "/rules/disable_test_001/disable",
            json={
                "actor": "admin_user",
                "reason": "Temporarily disabling rule",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rule"]["status"] == "disabled"

    def test_disable_from_shadow(self, client):
        """Test that can disable from shadow."""
        _create_and_submit_rule(client, "disable_test_002", actor="submitter_user")
        client.post(
            "/rules/draft/disable_test_002/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        # Move to shadow
        client.post(
            "/rules/disable_test_002/shadow",
            json={
                "actor": "admin_user",
                "reason": "Shadow mode",
            },
        )

        # Disable from shadow
        response = client.post(
            "/rules/disable_test_002/disable",
            json={
                "actor": "admin_user",
                "reason": "Disabling shadow rule",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rule"]["status"] == "disabled"

    def test_disable_changes_status(self, client):
        """Test that disable changes status to disabled."""
        _create_and_submit_rule(client, "disable_test_003", actor="submitter_user")
        client.post(
            "/rules/draft/disable_test_003/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        # Verify it's active
        get_response = client.get("/rules/draft/disable_test_003")
        assert get_response.json()["status"] == "active"

        # Disable it
        client.post(
            "/rules/disable_test_003/disable",
            json={
                "actor": "admin_user",
                "reason": "Disabling rule",
            },
        )

        # Verify status changed
        get_response = client.get("/rules/draft/disable_test_003")
        assert get_response.json()["status"] == "disabled"

    def test_disable_creates_version(self, client):
        """Test that disable creates a version snapshot."""
        _create_and_submit_rule(client, "disable_test_004", actor="submitter_user")
        client.post(
            "/rules/draft/disable_test_004/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        client.post(
            "/rules/disable_test_004/disable",
            json={
                "actor": "admin_user",
                "reason": "Disabling",
            },
        )

        version_store = get_version_store()
        versions = version_store.list_versions("disable_test_004")
        assert len(versions) >= 3  # At least create + approve + disable

        # Check latest version has disabled status
        latest = versions[-1]
        assert latest.rule.status == "disabled"

    def test_disable_creates_audit_record(self, client):
        """Test that disable creates an audit record."""
        _create_and_submit_rule(client, "disable_test_005", actor="submitter_user")
        client.post(
            "/rules/draft/disable_test_005/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        client.post(
            "/rules/disable_test_005/disable",
            json={
                "actor": "admin_user",
                "reason": "Disabling for maintenance",
            },
        )

        audit_logger = get_audit_logger()
        records = audit_logger.get_rule_history("disable_test_005")

        # Find disable state_change record
        disable_records = [
            r
            for r in records
            if r.action == "state_change" and r.after_state.get("status") == "disabled"
        ]
        assert len(disable_records) > 0

        disable_record = disable_records[-1]
        assert disable_record.actor == "admin_user"

    def test_disable_invalid_status_returns_400(self, client):
        """Test that disabling from invalid status returns 400."""
        # Create but don't submit (stays in draft)
        client.post(
            "/rules/draft",
            json={
                "id": "disable_test_006",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Test",
                "actor": "test_user",
            },
        )

        # Try to disable from draft (should fail)
        response = client.post(
            "/rules/disable_test_006/disable",
            json={
                "actor": "admin_user",
                "reason": "Cannot disable draft",
            },
        )
        assert response.status_code == 400
        assert "cannot disable" in response.json()["detail"].lower()


class TestShadowRule:
    """Tests for POST /rules/{rule_id}/shadow endpoint."""

    def test_shadow_from_active(self, client):
        """Test that can move from active to shadow."""
        _create_and_submit_rule(client, "shadow_test_001", actor="submitter_user")
        client.post(
            "/rules/draft/shadow_test_001/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        response = client.post(
            "/rules/shadow_test_001/shadow",
            json={
                "actor": "admin_user",
                "reason": "Moving to shadow for monitoring",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["rule"]["status"] == "shadow"

    def test_shadow_changes_status(self, client):
        """Test that shadow changes status to shadow."""
        _create_and_submit_rule(client, "shadow_test_002", actor="submitter_user")
        client.post(
            "/rules/draft/shadow_test_002/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        # Verify it's active
        get_response = client.get("/rules/draft/shadow_test_002")
        assert get_response.json()["status"] == "active"

        # Move to shadow
        client.post(
            "/rules/shadow_test_002/shadow",
            json={
                "actor": "admin_user",
                "reason": "Shadow mode",
            },
        )

        # Verify status changed
        get_response = client.get("/rules/draft/shadow_test_002")
        assert get_response.json()["status"] == "shadow"

    def test_shadow_creates_version(self, client):
        """Test that shadow creates a version snapshot."""
        _create_and_submit_rule(client, "shadow_test_003", actor="submitter_user")
        client.post(
            "/rules/draft/shadow_test_003/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        client.post(
            "/rules/shadow_test_003/shadow",
            json={
                "actor": "admin_user",
                "reason": "Shadow",
            },
        )

        version_store = get_version_store()
        versions = version_store.list_versions("shadow_test_003")
        assert len(versions) >= 3  # At least create + approve + shadow

        # Check latest version has shadow status
        latest = versions[-1]
        assert latest.rule.status == "shadow"

    def test_shadow_creates_audit_record(self, client):
        """Test that shadow creates an audit record."""
        _create_and_submit_rule(client, "shadow_test_004", actor="submitter_user")
        client.post(
            "/rules/draft/shadow_test_004/approve",
            json={
                "approver": "approver_user",  # Different from submitter_user
                "reason": "Approved",
            },
        )

        client.post(
            "/rules/shadow_test_004/shadow",
            json={
                "actor": "admin_user",
                "reason": "Moving to shadow mode",
            },
        )

        audit_logger = get_audit_logger()
        records = audit_logger.get_rule_history("shadow_test_004")

        # Find shadow state_change record
        shadow_records = [
            r
            for r in records
            if r.action == "state_change" and r.after_state.get("status") == "shadow"
        ]
        assert len(shadow_records) > 0

        shadow_record = shadow_records[-1]
        assert shadow_record.actor == "admin_user"

    def test_shadow_invalid_status_returns_400(self, client):
        """Test that shadowing from invalid status returns 400."""
        # Create but don't submit (stays in draft)
        client.post(
            "/rules/draft",
            json={
                "id": "shadow_test_005",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Test",
                "actor": "test_user",
            },
        )

        # Try to shadow from draft (should fail)
        response = client.post(
            "/rules/shadow_test_005/shadow",
            json={
                "actor": "admin_user",
                "reason": "Cannot shadow draft",
            },
        )
        assert response.status_code == 400
        assert "cannot shadow" in response.json()["detail"].lower()


class TestPublishRule:
    """Tests for POST /rules/{rule_id}/publish endpoint."""

    def test_publish_approved_rule_returns_200(self, client):
        """Test that publishing an approved rule returns 200."""
        _create_and_submit_rule(client, "publish_test_001", actor="submitter_user")
        # Approve it first
        client.post(
            "/rules/draft/publish_test_001/approve",
            json={
                "approver": "approver_user",
                "reason": "Approved",
            },
        )

        # Publish it
        response = client.post(
            "/rules/publish_test_001/publish",
            json={
                "actor": "operator_user",
                "reason": "Ready for production",
            },
        )
        assert response.status_code == 200
        assert response.json()["rule"]["status"] == "active"
        assert "published_at" in response.json()
        assert "version_id" in response.json()

    def test_publish_changes_status_to_active(self, client):
        """Test that publish changes status from approved to active."""
        _create_and_submit_rule(client, "publish_test_002")
        # Approve it
        client.post(
            "/rules/draft/publish_test_002/approve",
            json={
                "approver": "approver_user",
                "reason": "Approved",
            },
        )

        # Verify it's approved
        get_response = client.get("/rules/draft/publish_test_002")
        assert get_response.json()["status"] == "approved"

        # Publish it
        client.post(
            "/rules/publish_test_002/publish",
            json={
                "actor": "operator_user",
                "reason": "Publishing to production",
            },
        )

        # Verify status changed to active
        get_response = client.get("/rules/draft/publish_test_002")
        assert get_response.json()["status"] == "active"

    def test_publish_creates_audit_record(self, client):
        """Test that publish creates RULE_PUBLISHED audit record."""
        _create_and_submit_rule(client, "publish_test_003")
        # Approve it
        client.post(
            "/rules/draft/publish_test_003/approve",
            json={
                "approver": "approver_user",
                "reason": "Approved",
            },
        )

        # Publish it
        client.post(
            "/rules/publish_test_003/publish",
            json={
                "actor": "operator_user",
                "reason": "Publishing to production",
            },
        )

        audit_logger = get_audit_logger()
        records = audit_logger.get_rule_history("publish_test_003")

        # Find RULE_PUBLISHED record
        publish_records = [
            r for r in records if r.action == "RULE_PUBLISHED"
        ]
        assert len(publish_records) > 0

        publish_record = publish_records[-1]
        assert publish_record.actor == "operator_user"
        assert publish_record.before_state["status"] == "approved"
        assert publish_record.after_state["status"] == "active"

    def test_publish_requires_approved_status(self, client):
        """Test that publish requires rule to be in approved status."""
        _create_and_submit_rule(client, "publish_test_004")
        # Don't approve it - stays in pending_review

        # Try to publish (should fail)
        response = client.post(
            "/rules/publish_test_004/publish",
            json={
                "actor": "operator_user",
                "reason": "Cannot publish non-approved rule",
            },
        )
        assert response.status_code == 400
        assert "not approved" in response.json()["detail"].lower()

    def test_publish_not_found_returns_404(self, client):
        """Test that publishing non-existent rule returns 404."""
        response = client.post(
            "/rules/nonexistent/publish",
            json={
                "actor": "operator_user",
                "reason": "Test",
            },
        )
        assert response.status_code == 404

    def test_publish_requires_actor(self, client):
        """Test that publish requires actor field."""
        _create_and_submit_rule(client, "publish_test_005")
        # Approve it
        client.post(
            "/rules/draft/publish_test_005/approve",
            json={
                "approver": "approver_user",
                "reason": "Approved",
            },
        )

        # Try to publish without actor (should fail validation)
        response = client.post(
            "/rules/publish_test_005/publish",
            json={
                "reason": "No actor provided",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_approve_does_not_make_rule_active(self, client):
        """Regression test: approve should NOT make rule active."""
        _create_and_submit_rule(client, "publish_test_006")

        # Approve it
        client.post(
            "/rules/draft/publish_test_006/approve",
            json={
                "approver": "approver_user",
                "reason": "Approved",
            },
        )

        # Verify it's approved, NOT active
        get_response = client.get("/rules/draft/publish_test_006")
        assert get_response.json()["status"] == "approved"
        assert get_response.json()["status"] != "active"
