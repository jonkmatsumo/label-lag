"""Tests for rule version endpoints (list, get, rollback)."""

import pytest
from fastapi.testclient import TestClient

from api.audit import AuditLogger, get_audit_logger, set_audit_logger
from api.draft_store import DraftRuleStore, set_draft_store
from api.main import app
from api.versioning import (
    RuleVersionStore,
    set_version_store,
)


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


def _create_rule_with_versions(client, rule_id: str) -> None:
    """Helper to create a rule with multiple versions."""
    # Create rule (version 1)
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
            "reason": "Initial version",
            "actor": "user1",
        },
    )

    # Update rule (version 2)
    client.put(
        f"/rules/draft/{rule_id}",
        json={
            "value": 10,
            "reason": "Updated threshold",
            "actor": "user2",
        },
    )

    # Submit for review (version 3)
    client.post(
        f"/rules/draft/{rule_id}/submit",
        json={
            "actor": "user2",
            "justification": "Ready for review",
        },
    )


class TestListRuleVersions:
    """Tests for GET /rules/{rule_id}/versions endpoint."""

    def test_list_versions_returns_200(self, client):
        """Test that listing versions returns 200."""
        _create_rule_with_versions(client, "version_test_001")

        response = client.get("/rules/version_test_001/versions")
        assert response.status_code == 200

    def test_list_versions_response_structure(self, client):
        """Test that response has correct structure."""
        _create_rule_with_versions(client, "version_test_002")

        response = client.get("/rules/version_test_002/versions")
        data = response.json()

        assert "versions" in data
        assert "total" in data
        assert isinstance(data["versions"], list)
        assert data["total"] == len(data["versions"])

    def test_list_versions_ordered_by_timestamp(self, client):
        """Test that versions are ordered oldest first."""
        _create_rule_with_versions(client, "version_test_003")

        response = client.get("/rules/version_test_003/versions")
        data = response.json()

        versions = data["versions"]
        assert len(versions) >= 3

        # Check timestamps are in ascending order
        timestamps = [v["timestamp"] for v in versions]
        assert timestamps == sorted(timestamps)

    def test_list_versions_empty_returns_empty_list(self, client):
        """Test that no versions returns empty list."""
        # Create rule but don't create any versions via API
        # (versions are created on create/update/submit operations)
        response = client.get("/rules/nonexistent_rule/versions")
        assert response.status_code == 200
        data = response.json()
        assert data["versions"] == []
        assert data["total"] == 0

    def test_list_versions_includes_all_fields(self, client):
        """Test that all version fields are present."""
        _create_rule_with_versions(client, "version_test_004")

        response = client.get("/rules/version_test_004/versions")
        data = response.json()

        if len(data["versions"]) > 0:
            version = data["versions"][0]
            assert "rule_id" in version
            assert "version_id" in version
            assert "rule" in version
            assert "timestamp" in version
            assert "created_by" in version
            assert "reason" in version

            # Check rule snapshot is included
            rule = version["rule"]
            assert "rule_id" in rule
            assert "field" in rule
            assert "status" in rule


class TestGetRuleVersion:
    """Tests for GET /rules/{rule_id}/versions/{version_id} endpoint."""

    def test_get_version_returns_200(self, client):
        """Test that getting a version returns 200."""
        _create_rule_with_versions(client, "version_test_005")

        # Get list to find a version_id
        list_response = client.get("/rules/version_test_005/versions")
        versions = list_response.json()["versions"]
        assert len(versions) > 0

        version_id = versions[0]["version_id"]

        response = client.get(f"/rules/version_test_005/versions/{version_id}")
        assert response.status_code == 200

    def test_get_version_response_structure(self, client):
        """Test that response has correct structure."""
        _create_rule_with_versions(client, "version_test_006")

        # Get list to find a version_id
        list_response = client.get("/rules/version_test_006/versions")
        versions = list_response.json()["versions"]
        assert len(versions) > 0

        version_id = versions[0]["version_id"]

        response = client.get(f"/rules/version_test_006/versions/{version_id}")
        data = response.json()

        assert "rule_id" in data
        assert "version_id" in data
        assert "rule" in data
        assert "timestamp" in data
        assert "created_by" in data
        assert "reason" in data

    def test_get_version_not_found_returns_404(self, client):
        """Test that getting non-existent version returns 404."""
        _create_rule_with_versions(client, "version_test_007")

        response = client.get("/rules/version_test_007/versions/nonexistent_version_id")
        assert response.status_code == 404

    def test_get_version_includes_rule_snapshot(self, client):
        """Test that rule snapshot is included in response."""
        _create_rule_with_versions(client, "version_test_008")

        # Get list to find version_ids
        list_response = client.get("/rules/version_test_008/versions")
        versions = list_response.json()["versions"]

        # Get first version (should have value=5)
        first_version_id = versions[0]["version_id"]
        response = client.get(f"/rules/version_test_008/versions/{first_version_id}")
        data = response.json()

        rule = data["rule"]
        assert rule["rule_id"] == "version_test_008"
        assert rule["field"] == "velocity_24h"
        # First version should have original value
        assert rule["value"] == 5


class TestRollbackRuleVersion:
    """Tests for POST /rules/{rule_id}/versions/{version_id}/rollback endpoint."""

    def test_rollback_returns_200(self, client):
        """Test that rolling back returns 200."""
        _create_rule_with_versions(client, "rollback_test_001")

        # Get list to find first version
        list_response = client.get("/rules/rollback_test_001/versions")
        versions = list_response.json()["versions"]
        assert len(versions) >= 2

        target_version_id = versions[0]["version_id"]  # First (oldest) version

        response = client.post(
            f"/rules/rollback_test_001/versions/{target_version_id}/rollback",
            json={
                "actor": "admin_user",
                "reason": "Rolling back to previous version",
            },
        )
        assert response.status_code == 200

    def test_rollback_response_structure(self, client):
        """Test that response has correct structure."""
        _create_rule_with_versions(client, "rollback_test_002")

        # Get list to find first version
        list_response = client.get("/rules/rollback_test_002/versions")
        versions = list_response.json()["versions"]
        target_version_id = versions[0]["version_id"]

        response = client.post(
            f"/rules/rollback_test_002/versions/{target_version_id}/rollback",
            json={
                "actor": "admin_user",
                "reason": "Rollback test",
            },
        )
        data = response.json()

        assert "rule" in data
        assert "version_id" in data
        assert "rolled_back_to" in data
        assert "rolled_back_at" in data
        assert data["rolled_back_to"] == target_version_id

    def test_rollback_creates_new_version(self, client):
        """Test that rollback creates a new version (history preserved)."""
        _create_rule_with_versions(client, "rollback_test_003")

        # Get initial version count
        list_response = client.get("/rules/rollback_test_003/versions")
        initial_count = list_response.json()["total"]

        target_version_id = list_response.json()["versions"][0]["version_id"]

        # Rollback
        client.post(
            f"/rules/rollback_test_003/versions/{target_version_id}/rollback",
            json={
                "actor": "admin_user",
                "reason": "Rollback",
            },
        )

        # Check version count increased
        list_response = client.get("/rules/rollback_test_003/versions")
        new_count = list_response.json()["total"]
        assert new_count == initial_count + 1

    def test_rollback_rule_matches_target_version(self, client):
        """Test that rolled back rule matches target version."""
        _create_rule_with_versions(client, "rollback_test_004")

        # Get first version (should have value=5)
        list_response = client.get("/rules/rollback_test_004/versions")
        versions = list_response.json()["versions"]
        target_version = versions[0]  # First (oldest) version
        target_version_id = target_version["version_id"]
        target_value = target_version["rule"]["value"]

        # Current rule should have value=10 (from update)
        get_response = client.get("/rules/draft/rollback_test_004")
        current_value = get_response.json()["value"]
        assert current_value == 10  # Updated value

        # Rollback
        rollback_response = client.post(
            f"/rules/rollback_test_004/versions/{target_version_id}/rollback",
            json={
                "actor": "admin_user",
                "reason": "Rollback to original",
            },
        )

        # Check rule value matches target version
        rollback_data = rollback_response.json()
        assert rollback_data["rule"]["value"] == target_value

        # Verify in draft store too
        get_response = client.get("/rules/draft/rollback_test_004")
        assert get_response.json()["value"] == target_value

    def test_rollback_creates_audit_record(self, client):
        """Test that rollback creates an audit record."""
        _create_rule_with_versions(client, "rollback_test_005")

        list_response = client.get("/rules/rollback_test_005/versions")
        target_version_id = list_response.json()["versions"][0]["version_id"]

        client.post(
            f"/rules/rollback_test_005/versions/{target_version_id}/rollback",
            json={
                "actor": "admin_user",
                "reason": "Rollback audit test",
            },
        )

        audit_logger = get_audit_logger()
        records = audit_logger.get_rule_history("rollback_test_005")

        # Find rollback record
        rollback_records = [r for r in records if r.action == "rollback"]
        assert len(rollback_records) > 0

        rollback_record = rollback_records[-1]
        assert rollback_record.actor == "admin_user"
        assert "rolled_back_to" in rollback_record.after_state
        assert rollback_record.after_state["rolled_back_to"] == target_version_id

    def test_rollback_not_found_returns_404(self, client):
        """Test that rolling back to non-existent version returns 404."""
        _create_rule_with_versions(client, "rollback_test_006")

        response = client.post(
            "/rules/rollback_test_006/versions/nonexistent_version/rollback",
            json={
                "actor": "admin_user",
                "reason": "Test",
            },
        )
        assert response.status_code == 404

    def test_rollback_updates_draft_store(self, client):
        """Test that rollback updates draft store if rule exists."""
        _create_rule_with_versions(client, "rollback_test_007")

        # Get first version
        list_response = client.get("/rules/rollback_test_007/versions")
        target_version_id = list_response.json()["versions"][0]["version_id"]
        target_value = list_response.json()["versions"][0]["rule"]["value"]

        # Rollback
        client.post(
            f"/rules/rollback_test_007/versions/{target_version_id}/rollback",
            json={
                "actor": "admin_user",
                "reason": "Rollback",
            },
        )

        # Verify draft store updated
        get_response = client.get("/rules/draft/rollback_test_007")
        assert get_response.json()["value"] == target_value


class TestRuleDiffEndpoint:
    """Tests for GET /rules/{rule_id}/diff endpoint."""

    def test_diff_200_both_versions_specified(self, client):
        """Test diff with both versions explicitly specified."""
        _create_rule_with_versions(client, "diff_test_001")

        # Get versions
        list_response = client.get("/rules/diff_test_001/versions")
        versions = list_response.json()["versions"]
        assert len(versions) >= 2

        version_a = versions[-1]["version_id"]  # Latest
        version_b = versions[0]["version_id"]  # Oldest

        response = client.get(
            f"/rules/diff_test_001/diff?version_a={version_a}&version_b={version_b}"
        )
        assert response.status_code == 200

    def test_diff_200_no_params_latest_vs_previous(self, client):
        """Test diff with no params defaults to latest vs previous."""
        _create_rule_with_versions(client, "diff_test_002")

        # Get versions for verification
        list_response = client.get("/rules/diff_test_002/versions")
        versions = list_response.json()["versions"]
        assert len(versions) >= 2

        latest = versions[-1]["version_id"]
        previous = versions[-2]["version_id"]

        response = client.get("/rules/diff_test_002/diff")
        assert response.status_code == 200

        data = response.json()
        assert data["version_a_id"] == latest
        assert data["version_b_id"] == previous

    def test_diff_200_only_version_a_uses_predecessor(self, client):
        """Test diff with only version_a uses predecessor."""
        _create_rule_with_versions(client, "diff_test_003")

        list_response = client.get("/rules/diff_test_003/versions")
        versions = list_response.json()["versions"]
        assert len(versions) >= 2

        # Use second version (has a predecessor)
        version_a = versions[1]["version_id"]
        expected_b = versions[0]["version_id"]

        response = client.get(f"/rules/diff_test_003/diff?version_a={version_a}")
        assert response.status_code == 200

        data = response.json()
        assert data["version_a_id"] == version_a
        assert data["version_b_id"] == expected_b

    def test_diff_200_only_version_b_compares_to_latest(self, client):
        """Test diff with only version_b compares latest to version_b."""
        _create_rule_with_versions(client, "diff_test_004")

        list_response = client.get("/rules/diff_test_004/versions")
        versions = list_response.json()["versions"]
        assert len(versions) >= 2

        version_b = versions[0]["version_id"]  # Oldest
        expected_a = versions[-1]["version_id"]  # Latest

        response = client.get(f"/rules/diff_test_004/diff?version_b={version_b}")
        assert response.status_code == 200

        data = response.json()
        assert data["version_a_id"] == expected_a
        assert data["version_b_id"] == version_b

    def test_diff_400_same_version(self, client):
        """Test diff with same version returns 400."""
        _create_rule_with_versions(client, "diff_test_005")

        list_response = client.get("/rules/diff_test_005/versions")
        versions = list_response.json()["versions"]
        version_id = versions[0]["version_id"]

        response = client.get(
            f"/rules/diff_test_005/diff?version_a={version_id}&version_b={version_id}"
        )
        assert response.status_code == 400
        assert "Cannot compare a version to itself" in response.json()["detail"]

    def test_diff_400_insufficient_versions(self, client):
        """Test diff when only one version exists returns 400."""
        # Create rule with only one version (no update/submit)
        client.post(
            "/rules/draft",
            json={
                "id": "diff_test_006",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Single version",
                "actor": "user1",
            },
        )

        # Try to get diff without params (needs predecessor)
        response = client.get("/rules/diff_test_006/diff")
        assert response.status_code == 400
        assert "no predecessor" in response.json()["detail"]

    def test_diff_404_rule_not_found(self, client):
        """Test diff for nonexistent rule returns 404."""
        response = client.get("/rules/nonexistent_rule/diff")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_diff_404_version_a_not_found(self, client):
        """Test diff with nonexistent version_a returns 404."""
        _create_rule_with_versions(client, "diff_test_007")

        response = client.get(
            "/rules/diff_test_007/diff?version_a=nonexistent_version"
        )
        assert response.status_code == 404
        assert "Version not found" in response.json()["detail"]

    def test_diff_404_version_b_not_found(self, client):
        """Test diff with nonexistent version_b returns 404."""
        _create_rule_with_versions(client, "diff_test_008")

        list_response = client.get("/rules/diff_test_008/versions")
        version_a = list_response.json()["versions"][-1]["version_id"]

        response = client.get(
            f"/rules/diff_test_008/diff?version_a={version_a}"
            "&version_b=nonexistent_version"
        )
        assert response.status_code == 404
        assert "Version not found" in response.json()["detail"]

    def test_diff_response_structure(self, client):
        """Test diff response has all required fields."""
        _create_rule_with_versions(client, "diff_test_009")

        response = client.get("/rules/diff_test_009/diff")
        assert response.status_code == 200

        data = response.json()
        assert "version_a_id" in data
        assert "version_b_id" in data
        assert "rule_id" in data
        assert "changes" in data
        assert "is_breaking" in data
        assert "version_a_timestamp" in data
        assert "version_b_timestamp" in data
        assert "version_a_created_by" in data
        assert "version_b_created_by" in data

        # Verify changes structure
        assert isinstance(data["changes"], list)
        if len(data["changes"]) > 0:
            change = data["changes"][0]
            assert "field_name" in change
            assert "change_type" in change
            assert "old_value" in change
            assert "new_value" in change

    def test_diff_breaking_flag_correct(self, client):
        """Test breaking flag is set correctly for field/op/action changes."""
        # Create rule and update with value change only (not breaking)
        _create_rule_with_versions(client, "diff_test_010")

        response = client.get("/rules/diff_test_010/diff")
        data = response.json()

        # Value change is NOT breaking (only field/op/action are breaking)
        # Check that is_breaking reflects the actual changes
        value_change = next(
            (c for c in data["changes"] if c["field_name"] == "value"), None
        )
        if value_change and value_change["change_type"] == "modified":
            # If only value changed, is_breaking should be False
            field_change = next(
                (c for c in data["changes"] if c["field_name"] == "field"), None
            )
            op_change = next(
                (c for c in data["changes"] if c["field_name"] == "op"), None
            )
            action_change = next(
                (c for c in data["changes"] if c["field_name"] == "action"), None
            )

            breaking_fields_changed = any(
                c and c["change_type"] == "modified"
                for c in [field_change, op_change, action_change]
            )

            assert data["is_breaking"] == breaking_fields_changed

    def test_diff_all_fields_included(self, client):
        """Test that diff includes all rule fields."""
        _create_rule_with_versions(client, "diff_test_011")

        response = client.get("/rules/diff_test_011/diff")
        data = response.json()

        expected_fields = {
            "field", "op", "value", "action", "score",
            "severity", "reason", "status"
        }
        actual_fields = {c["field_name"] for c in data["changes"]}
        assert expected_fields == actual_fields
