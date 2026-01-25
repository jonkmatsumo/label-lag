"""Tests for draft rule API endpoints."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.draft_store import DraftRuleStore, get_draft_store, set_draft_store
from api.main import app
from api.rules import Rule, RuleStatus


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def draft_store():
    """Create a test draft store."""
    store = DraftRuleStore()
    yield store
    set_draft_store(DraftRuleStore())  # Reset global store


@pytest.fixture(autouse=True)
def reset_draft_store():
    """Reset draft store before each test."""
    set_draft_store(DraftRuleStore())
    yield
    set_draft_store(DraftRuleStore())


class TestCreateDraftRule:
    """Tests for POST /rules/draft endpoint."""

    def test_create_draft_rule_returns_200(self, client):
        """Test that creating a draft rule returns 200."""
        response = client.post(
            "/rules/draft",
            json={
                "id": "test_rule_001",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Test rule",
                "actor": "test_user",
            },
        )
        assert response.status_code == 200

    def test_create_draft_rule_response_structure(self, client):
        """Test that response has correct structure."""
        response = client.post(
            "/rules/draft",
            json={
                "id": "test_rule_002",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Test rule",
                "actor": "test_user",
            },
        )
        data = response.json()

        assert "rule_id" in data
        assert "rule" in data
        assert "validation" in data
        assert "created_at" in data

        rule = data["rule"]
        assert rule["rule_id"] == "test_rule_002"
        assert rule["status"] == "draft"

    def test_create_draft_rule_enforces_draft_status(self, client):
        """Test that created rule always has draft status."""
        response = client.post(
            "/rules/draft",
            json={
                "id": "test_rule_003",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Test rule",
                "actor": "test_user",
            },
        )
        data = response.json()
        assert data["rule"]["status"] == "draft"

    def test_create_draft_rule_duplicate_id_returns_409(self, client):
        """Test that creating duplicate rule ID returns 409."""
        # Create first rule
        client.post(
            "/rules/draft",
            json={
                "id": "duplicate_rule",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "First rule",
                "actor": "test_user",
            },
        )

        # Try to create duplicate
        response = client.post(
            "/rules/draft",
            json={
                "id": "duplicate_rule",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Duplicate rule",
                "actor": "test_user",
            },
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_create_draft_rule_invalid_operator_returns_400(self, client):
        """Test that invalid operator returns 400."""
        response = client.post(
            "/rules/draft",
            json={
                "id": "invalid_rule",
                "field": "velocity_24h",
                "op": "invalid_op",
                "value": 5,
                "action": "clamp_min",
                "score": 70,
                "severity": "medium",
                "reason": "Invalid",
                "actor": "test_user",
            },
        )
        assert response.status_code == 400

    def test_create_draft_rule_missing_score_for_action_returns_400(self, client):
        """Test that missing score for score-requiring action returns 400."""
        response = client.post(
            "/rules/draft",
            json={
                "id": "missing_score",
                "field": "velocity_24h",
                "op": ">",
                "value": 5,
                "action": "clamp_min",
                "score": None,  # Missing score
                "severity": "medium",
                "reason": "Missing score",
                "actor": "test_user",
            },
        )
        assert response.status_code == 400

    def test_create_draft_rule_validation_included(self, client):
        """Test that validation results are included in response."""
        response = client.post(
            "/rules/draft",
            json={
                "id": "validated_rule",
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
        data = response.json()

        assert "validation" in data
        validation = data["validation"]
        assert "conflicts" in validation
        assert "redundancies" in validation
        assert "is_valid" in validation
        assert isinstance(validation["conflicts"], list)
        assert isinstance(validation["redundancies"], list)
        assert isinstance(validation["is_valid"], bool)


class TestListDraftRules:
    """Tests for GET /rules/draft endpoint."""

    def test_list_draft_rules_returns_200(self, client):
        """Test that listing draft rules returns 200."""
        response = client.get("/rules/draft")
        assert response.status_code == 200

    def test_list_draft_rules_response_structure(self, client):
        """Test that response has correct structure."""
        # Create a draft rule first
        client.post(
            "/rules/draft",
            json={
                "id": "list_test_001",
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

        response = client.get("/rules/draft")
        data = response.json()

        assert "rules" in data
        assert "total" in data
        assert isinstance(data["rules"], list)
        assert isinstance(data["total"], int)

    def test_list_draft_rules_includes_created_rule(self, client):
        """Test that created rule appears in list."""
        # Create a rule
        create_response = client.post(
            "/rules/draft",
            json={
                "id": "list_test_002",
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
        created_id = create_response.json()["rule_id"]

        # List rules
        list_response = client.get("/rules/draft")
        rules = list_response.json()["rules"]

        rule_ids = [r["rule_id"] for r in rules]
        assert created_id in rule_ids

    def test_list_draft_rules_excludes_archived_by_default(self, client):
        """Test that archived rules are excluded by default."""
        # Create a rule
        client.post(
            "/rules/draft",
            json={
                "id": "archived_test",
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

        # Archive it
        store = get_draft_store()
        rule = store.get("archived_test")
        if rule:
            rule_dict = rule.__dict__.copy()
            rule_dict["status"] = RuleStatus.ARCHIVED.value
            archived_rule = Rule(**rule_dict)
            store._rules["archived_test"] = archived_rule

        # List rules (should exclude archived)
        response = client.get("/rules/draft")
        rules = response.json()["rules"]

        rule_ids = [r["rule_id"] for r in rules]
        assert "archived_test" not in rule_ids

    def test_list_draft_rules_with_status_filter(self, client):
        """Test listing with status filter."""
        # Create a rule
        client.post(
            "/rules/draft",
            json={
                "id": "status_filter_test",
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

        # List with draft status filter
        response = client.get("/rules/draft?status=draft")
        rules = response.json()["rules"]

        assert all(r["status"] == "draft" for r in rules)


class TestGetDraftRule:
    """Tests for GET /rules/draft/{rule_id} endpoint."""

    def test_get_draft_rule_returns_200(self, client):
        """Test that getting a draft rule returns 200."""
        # Create a rule first
        client.post(
            "/rules/draft",
            json={
                "id": "get_test_001",
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

        response = client.get("/rules/draft/get_test_001")
        assert response.status_code == 200

    def test_get_draft_rule_response_structure(self, client):
        """Test that response has correct structure."""
        # Create a rule
        create_response = client.post(
            "/rules/draft",
            json={
                "id": "get_test_002",
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

        # Get the rule
        response = client.get("/rules/draft/get_test_002")
        data = response.json()

        assert "rule_id" in data
        assert "field" in data
        assert "op" in data
        assert "value" in data
        assert "action" in data
        assert "status" in data
        assert data["rule_id"] == "get_test_002"
        assert data["status"] == "draft"

    def test_get_draft_rule_not_found_returns_404(self, client):
        """Test that getting non-existent rule returns 404."""
        response = client.get("/rules/draft/nonexistent_rule")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestDraftRuleStore:
    """Tests for DraftRuleStore class."""

    def test_save_draft_rule(self, draft_store):
        """Test saving a draft rule."""
        rule = Rule(
            id="store_test_001",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            severity="medium",
            reason="Test",
            status=RuleStatus.DRAFT.value,
        )

        draft_store.save(rule)

        retrieved = draft_store.get("store_test_001")
        assert retrieved is not None
        assert retrieved.id == "store_test_001"
        assert retrieved.status == RuleStatus.DRAFT.value

    def test_save_non_draft_rule_raises(self, draft_store):
        """Test that saving non-draft rule raises ValueError."""
        rule = Rule(
            id="store_test_002",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            severity="medium",
            reason="Test",
            status=RuleStatus.ACTIVE.value,  # Not draft
        )

        with pytest.raises(ValueError, match="Only draft rules"):
            draft_store.save(rule)

    def test_list_rules_excludes_archived(self, draft_store):
        """Test that list_rules excludes archived by default."""
        # Create draft rule
        draft_rule = Rule(
            id="draft_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        draft_store.save(draft_rule)

        # Create archived rule
        archived_rule = Rule(
            id="archived_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.ARCHIVED.value,
        )
        draft_store._rules["archived_rule"] = archived_rule

        # List should exclude archived
        rules = draft_store.list_rules(include_archived=False)
        rule_ids = [r.id for r in rules]
        assert "draft_rule" in rule_ids
        assert "archived_rule" not in rule_ids

    def test_list_rules_includes_archived_when_requested(self, draft_store):
        """Test that list_rules includes archived when requested."""
        # Create draft and archived rules
        draft_rule = Rule(
            id="draft_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        draft_store.save(draft_rule)

        archived_rule = Rule(
            id="archived_rule",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.ARCHIVED.value,
        )
        draft_store._rules["archived_rule"] = archived_rule

        # List should include archived
        rules = draft_store.list_rules(include_archived=True)
        rule_ids = [r.id for r in rules]
        assert "draft_rule" in rule_ids
        assert "archived_rule" in rule_ids

    def test_delete_archives_rule(self, draft_store):
        """Test that delete archives the rule."""
        rule = Rule(
            id="delete_test",
            field="velocity_24h",
            op=">",
            value=5,
            action="clamp_min",
            score=70,
            status=RuleStatus.DRAFT.value,
        )
        draft_store.save(rule)

        # Delete (archive)
        result = draft_store.delete("delete_test")
        assert result is True

        # Rule should be archived
        archived = draft_store.get("delete_test")
        assert archived is not None
        assert archived.status == RuleStatus.ARCHIVED.value

    def test_delete_non_existent_returns_false(self, draft_store):
        """Test that deleting non-existent rule returns False."""
        result = draft_store.delete("nonexistent")
        assert result is False
