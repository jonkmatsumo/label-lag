"""Tests for audit log endpoints (query, history, export)."""

import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from api.audit import AuditLogger, set_audit_logger
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


def _create_audit_events(client):
    """Helper to create multiple audit events."""
    # Create rule 1
    client.post(
        "/rules/draft",
        json={
            "id": "audit_rule_1",
            "field": "velocity_24h",
            "op": ">",
            "value": 5,
            "action": "clamp_min",
            "score": 70,
            "severity": "medium",
            "reason": "Test rule 1",
            "actor": "user1",
        },
    )

    # Create rule 2
    client.post(
        "/rules/draft",
        json={
            "id": "audit_rule_2",
            "field": "amount_to_avg_ratio_30d",
            "op": ">",
            "value": 3.0,
            "action": "clamp_min",
            "score": 75,
            "severity": "high",
            "reason": "Test rule 2",
            "actor": "user2",
        },
    )

    # Update rule 1
    client.put(
        "/rules/draft/audit_rule_1",
        json={
            "value": 10,
            "reason": "Updated rule 1",
            "actor": "user1",
        },
    )

    # Submit rule 1
    client.post(
        "/rules/draft/audit_rule_1/submit",
        json={
            "actor": "user1",
            "justification": "Submitting rule 1 for review",
        },
    )


class TestQueryAuditLogs:
    """Tests for GET /audit/logs endpoint."""

    def test_query_audit_logs_returns_200(self, client):
        """Test that querying audit logs returns 200."""
        _create_audit_events(client)

        response = client.get("/audit/logs")
        assert response.status_code == 200

    def test_query_audit_logs_response_structure(self, client):
        """Test that response has correct structure."""
        _create_audit_events(client)

        response = client.get("/audit/logs")
        data = response.json()

        assert "records" in data
        assert "total" in data
        assert isinstance(data["records"], list)
        assert data["total"] == len(data["records"])

        if len(data["records"]) > 0:
            record = data["records"][0]
            assert "rule_id" in record
            assert "action" in record
            assert "actor" in record
            assert "timestamp" in record

    def test_query_by_rule_id(self, client):
        """Test that filtering by rule_id works."""
        _create_audit_events(client)

        response = client.get("/audit/logs", params={"rule_id": "audit_rule_1"})
        data = response.json()

        assert all(r["rule_id"] == "audit_rule_1" for r in data["records"])
        assert data["total"] > 0

    def test_query_by_actor(self, client):
        """Test that filtering by actor works."""
        _create_audit_events(client)

        response = client.get("/audit/logs", params={"actor": "user1"})
        data = response.json()

        assert all(r["actor"] == "user1" for r in data["records"])
        assert data["total"] > 0

    def test_query_by_action(self, client):
        """Test that filtering by action works."""
        _create_audit_events(client)

        response = client.get("/audit/logs", params={"action": "create"})
        data = response.json()

        assert all(r["action"] == "create" for r in data["records"])
        assert data["total"] > 0

    def test_query_by_date_range(self, client):
        """Test that date range filtering works."""
        _create_audit_events(client)

        # Get all records to find date range
        all_response = client.get("/audit/logs")
        all_records = all_response.json()["records"]
        assert len(all_records) > 0

        # Use timestamps from records
        timestamps = [r["timestamp"] for r in all_records]
        start_date = min(timestamps)
        end_date = max(timestamps)

        # Query with date range
        response = client.get(
            "/audit/logs",
            params={"start_date": start_date, "end_date": end_date},
        )
        data = response.json()

        # Should include all records
        assert data["total"] == len(all_records)

    def test_query_combines_filters(self, client):
        """Test that multiple filters work together."""
        _create_audit_events(client)

        response = client.get(
            "/audit/logs",
            params={
                "rule_id": "audit_rule_1",
                "actor": "user1",
                "action": "create",
            },
        )
        data = response.json()

        for record in data["records"]:
            assert record["rule_id"] == "audit_rule_1"
            assert record["actor"] == "user1"
            assert record["action"] == "create"

    def test_query_ordered_by_timestamp(self, client):
        """Test that results are ordered by timestamp (oldest first)."""
        _create_audit_events(client)

        response = client.get("/audit/logs")
        data = response.json()

        records = data["records"]
        if len(records) > 1:
            timestamps = [r["timestamp"] for r in records]
            assert timestamps == sorted(timestamps)


class TestGetRuleAuditHistory:
    """Tests for GET /audit/rules/{rule_id}/history endpoint."""

    def test_get_rule_history_returns_200(self, client):
        """Test that getting rule history returns 200."""
        _create_audit_events(client)

        response = client.get("/audit/rules/audit_rule_1/history")
        assert response.status_code == 200

    def test_get_rule_history_response_structure(self, client):
        """Test that response has correct structure."""
        _create_audit_events(client)

        response = client.get("/audit/rules/audit_rule_1/history")
        data = response.json()

        assert "records" in data
        assert "total" in data
        assert isinstance(data["records"], list)
        assert data["total"] == len(data["records"])

    def test_get_rule_history_includes_all_events(self, client):
        """Test that all events for the rule are included."""
        _create_audit_events(client)

        response = client.get("/audit/rules/audit_rule_1/history")
        data = response.json()

        # Should have at least create, update, and submit events
        assert data["total"] >= 3

        actions = [r["action"] for r in data["records"]]
        assert "create" in actions
        assert "update" in actions
        assert "state_change" in actions

    def test_get_rule_history_ordered_by_timestamp(self, client):
        """Test that events are ordered chronologically."""
        _create_audit_events(client)

        response = client.get("/audit/rules/audit_rule_1/history")
        data = response.json()

        records = data["records"]
        if len(records) > 1:
            timestamps = [r["timestamp"] for r in records]
            assert timestamps == sorted(timestamps)


class TestExportAuditLogs:
    """Tests for GET /audit/export endpoint."""

    def test_export_json_returns_200(self, client):
        """Test that JSON export returns 200."""
        _create_audit_events(client)

        response = client.get("/audit/export", params={"format": "json"})
        assert response.status_code == 200

    def test_export_json_content_type(self, client):
        """Test that JSON export has correct Content-Type."""
        _create_audit_events(client)

        response = client.get("/audit/export", params={"format": "json"})
        assert response.headers["content-type"] == "application/json"

    def test_export_json_structure(self, client):
        """Test that JSON structure is valid."""
        _create_audit_events(client)

        response = client.get("/audit/export", params={"format": "json"})
        data = json.loads(response.text)

        assert isinstance(data, list)
        if len(data) > 0:
            record = data[0]
            assert "rule_id" in record
            assert "action" in record
            assert "actor" in record
            assert "timestamp" in record

    def test_export_csv_returns_200(self, client):
        """Test that CSV export returns 200."""
        _create_audit_events(client)

        response = client.get("/audit/export", params={"format": "csv"})
        assert response.status_code == 200

    def test_export_csv_content_type(self, client):
        """Test that CSV export has correct Content-Type."""
        _create_audit_events(client)

        response = client.get("/audit/export", params={"format": "csv"})
        # FastAPI may include charset in Content-Type
        assert response.headers["content-type"].startswith("text/csv")

    def test_export_csv_has_header(self, client):
        """Test that CSV has header row."""
        _create_audit_events(client)

        response = client.get("/audit/export", params={"format": "csv"})
        csv_content = response.text

        # Parse CSV
        reader = csv.reader(io.StringIO(csv_content))
        header = next(reader)

        expected_headers = [
            "rule_id",
            "action",
            "actor",
            "timestamp",
            "before_state",
            "after_state",
            "reason",
        ]
        assert header == expected_headers

    def test_export_respects_filters(self, client):
        """Test that filters are applied to export."""
        _create_audit_events(client)

        # Export with filter
        response = client.get(
            "/audit/export",
            params={"format": "json", "rule_id": "audit_rule_1"},
        )
        data = json.loads(response.text)

        assert all(r["rule_id"] == "audit_rule_1" for r in data)

    def test_export_invalid_format_returns_400(self, client):
        """Test that invalid format returns 400."""
        _create_audit_events(client)

        response = client.get("/audit/export", params={"format": "xml"})
        assert response.status_code == 400
        assert "invalid format" in response.json()["detail"].lower()
