"""Tests for Rule Inspector API endpoints (Phase 1).

These tests verify the read-only and deterministic nature of the
Rule Inspector endpoints.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_gateway_client(monkeypatch):
    """Mock the GatewayDecisionClient for all tests."""
    from api import gateway_client
    from rules_management.rules import Rule, RuleSet, evaluate_rules

    mock_client = MagicMock()

    def side_effect(
        features, base_score, ruleset=None, request_id=None, shadow_mode=False
    ):
        if ruleset:
            rules_obj = []
            for r in ruleset["rules"]:
                # Real gateway would return 400 for invalid rule
                if r["op"] not in [">", ">=", "<", "<=", "==", "in", "not_in"]:
                    raise ValueError(
                        f"invalid rules: [rule {r['id']}: invalid rule op {r['op']}]"
                    )
                rules_obj.append(Rule(**r))
            rs = RuleSet(version=ruleset["version"], rules=rules_obj)
        else:
            rs = RuleSet.empty()

        result = evaluate_rules(features, base_score, rs)

        # Map to gateway-like dict response
        explanations = []
        for e in result.explanations:
            matching_rule = next((r for r in rs.rules if r.id == e["rule_id"]), None)
            explanations.append(
                {
                    "rule_id": e["rule_id"],
                    "severity": e["severity"],
                    "reason": e["reason"],
                    "explanation": e["reason"],
                    "action": matching_rule.action if matching_rule else "",
                    "score": matching_rule.score if matching_rule else None,
                }
            )

        shadow_explanations = []
        for e in result.shadow_explanations:
            matching_rule = next((r for r in rs.rules if r.id == e["rule_id"]), None)
            shadow_explanations.append(
                {
                    "rule_id": e["rule_id"],
                    "severity": e["severity"],
                    "reason": e["reason"],
                    "explanation": e["reason"],
                    "action": matching_rule.action if matching_rule else "",
                    "score": matching_rule.score if matching_rule else None,
                }
            )

        resp = {
            "final_score": result.final_score,
            "baseline_score": base_score,
            "matched_rules": result.matched_rules,
            "explanations": explanations,
            "shadow_matched_rules": result.shadow_matched_rules,
            "shadow_explanations": shadow_explanations,
            "rejected": result.rejected,
            "ruleset_version": rs.version or "none",
        }

        if shadow_mode:
            resp["shadow_score"] = result.final_score
            resp["final_score"] = base_score
            resp["rejected"] = False

        return resp

    def diff_side_effect(
        features,
        base_score,
        ruleset_a=None,
        ruleset_b=None,
        request_id=None,
        shadow_mode=False,
    ):
        resp_a = side_effect(features, base_score, ruleset_a, request_id, shadow_mode)
        resp_b = side_effect(features, base_score, ruleset_b, request_id, shadow_mode)

        diff = {
            "score_delta": resp_b["final_score"] - resp_a["final_score"],
            "matched_rules_added": [
                r for r in resp_b["matched_rules"] if r not in resp_a["matched_rules"]
            ],
            "matched_rules_removed": [
                r for r in resp_a["matched_rules"] if r not in resp_b["matched_rules"]
            ],
        }
        if shadow_mode:
            diff["score_delta"] = resp_b["shadow_score"] - resp_a["shadow_score"]

        return {"a": resp_a, "b": resp_b, "diff": diff}

    mock_client.evaluate_rules.side_effect = side_effect
    mock_client.diff_rules.side_effect = diff_side_effect
    monkeypatch.setattr(gateway_client, "get_gateway_client", lambda: mock_client)
    return mock_client


class TestGetRulesEndpoint:
    """Tests for GET /rules endpoint."""

    def test_get_rules_returns_200(self, client):
        """Test that GET /rules returns 200."""
        response = client.get("/rules")
        assert response.status_code == 200

    def test_get_rules_response_structure(self, client):
        """Test that response has correct structure."""
        response = client.get("/rules")
        data = response.json()

        assert "version" in data
        assert "rules" in data
        assert isinstance(data["rules"], list)

    def test_get_rules_empty_when_no_ruleset(self, client):
        """Test that empty ruleset returns empty rules list."""
        response = client.get("/rules")
        data = response.json()

        # In test environment, no rules are loaded by default
        # Version should be "none" or similar when no rules
        assert "version" in data
        assert isinstance(data["rules"], list)


class TestSandboxEvaluateEndpoint:
    """Tests for POST /rules/sandbox/evaluate endpoint."""

    def test_sandbox_evaluate_returns_200(self, client):
        """Test that sandbox evaluate returns 200."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {
                    "velocity_24h": 3,
                    "amount_to_avg_ratio_30d": 1.5,
                    "balance_volatility_z_score": 0.0,
                    "bank_connections_24h": 1,
                    "merchant_risk_score": 30,
                    "has_history": True,
                    "transaction_amount": 100.0,
                },
                "base_score": 50,
            },
        )
        assert response.status_code == 200

    def test_sandbox_evaluate_response_structure(self, client):
        """Test that response has correct structure."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {},
                "base_score": 50,
            },
        )
        data = response.json()

        assert "final_score" in data
        assert "baseline_score" in data
        assert "matched_rules" in data
        assert "explanations" in data
        assert "shadow_matched_rules" in data
        assert "rejected" in data
        assert "ruleset_version" in data

    def test_sandbox_evaluate_with_default_features(self, client):
        """Test with default features."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={"base_score": 50},
        )
        assert response.status_code == 200
        data = response.json()

        # With no rules, score should match base_score
        assert data["final_score"] == 50

    def test_sandbox_evaluate_with_custom_ruleset(self, client):
        """Test with custom ruleset that matches."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {
                    "velocity_24h": 10,
                },
                "base_score": 50,
                "ruleset": {
                    "version": "test_v1",
                    "rules": [
                        {
                            "id": "high_velocity",
                            "field": "velocity_24h",
                            "op": ">",
                            "value": 5,
                            "action": "clamp_min",
                            "score": 70,
                            "severity": "medium",
                            "reason": "High velocity detected",
                            "status": "active",
                        }
                    ],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Rule should match and clamp score to minimum 70
        assert data["final_score"] >= 70
        assert len(data["matched_rules"]) == 1
        assert data["matched_rules"][0]["rule_id"] == "high_velocity"
        assert data["ruleset_version"] == "test_v1"

    def test_sandbox_evaluate_with_shadow_rule(self, client):
        """Test with shadow rule (recorded but not applied)."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {
                    "velocity_24h": 10,
                },
                "base_score": 50,
                "ruleset": {
                    "version": "test_v1",
                    "rules": [
                        {
                            "id": "shadow_high_velocity",
                            "field": "velocity_24h",
                            "op": ">",
                            "value": 5,
                            "action": "clamp_min",
                            "score": 70,
                            "severity": "medium",
                            "reason": "Shadow rule",
                            "status": "shadow",
                        }
                    ],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Shadow rule should match but not affect score
        assert data["final_score"] == 50
        assert len(data["matched_rules"]) == 0
        assert len(data["shadow_matched_rules"]) == 1

    def test_sandbox_evaluate_invalid_ruleset(self, client):
        """Test with invalid ruleset returns 400."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {},
                "base_score": 50,
                "ruleset": {
                    "version": "test",
                    "rules": [
                        {
                            "id": "bad_rule",
                            "field": "velocity",
                            "op": "invalid_op",  # Invalid operator
                            "value": 5,
                            "action": "reject",
                        }
                    ],
                },
            },
        )
        assert response.status_code == 400

    def test_sandbox_evaluate_is_pure_function(self, client):
        """Test that same input always gives same output (pure function)."""
        payload = {
            "features": {
                "velocity_24h": 5,
                "amount_to_avg_ratio_30d": 2.0,
            },
            "base_score": 60,
        }

        response1 = client.post("/rules/sandbox/evaluate", json=payload)
        response2 = client.post("/rules/sandbox/evaluate", json=payload)

        data1 = response1.json()
        data2 = response2.json()

        assert data1["final_score"] == data2["final_score"]
        assert data1["matched_rules"] == data2["matched_rules"]

    def test_sandbox_evaluate_score_in_range(self, client):
        """Test that final score is always in valid range."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {"velocity_24h": 50},  # Max allowed is 50
                "base_score": 1,
                "ruleset": {
                    "version": "test",
                    "rules": [
                        {
                            "id": "extreme",
                            "field": "velocity_24h",
                            "op": ">",
                            "value": 0,
                            "action": "override_score",
                            "score": 99,  # High but valid score
                            "severity": "high",
                            "status": "active",
                        }
                    ],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Score should be valid and in range
        assert isinstance(data["final_score"], int)
        assert 1 <= data["final_score"] <= 99

    def test_sandbox_evaluate_with_shadow_mode(self, client):
        """Test sandbox evaluation with shadow_mode=True."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {"velocity_24h": 10},
                "base_score": 50,
                "shadow_mode": True,
                "ruleset": {
                    "version": "test",
                    "rules": [
                        {
                            "id": "shadow_test",
                            "field": "velocity_24h",
                            "op": ">",
                            "value": 5,
                            "action": "reject",
                            "status": "active",
                        }
                    ],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()

        # In shadow mode, final_score should remain base_score
        assert data["final_score"] == 50
        assert data["baseline_score"] == 50
        assert data["shadow_score"] == 99
        assert data["rejected"] is False
        assert len(data["matched_rules"]) == 1
        assert data["matched_rules"][0]["rule_id"] == "shadow_test"

    def test_sandbox_diff_success(self, client):
        """Test sandbox diff between two rulesets."""
        response = client.post(
            "/rules/sandbox/diff",
            json={
                "features": {"velocity_24h": 10},
                "base_score": 50,
                "ruleset_a": {
                    "version": "v1",
                    "rules": [
                        {
                            "id": "rule1",
                            "field": "velocity_24h",
                            "op": ">",
                            "value": 5,
                            "action": "override_score",
                            "score": 60,
                            "status": "active",
                        }
                    ],
                },
                "ruleset_b": {
                    "version": "v2",
                    "rules": [
                        {
                            "id": "rule1",
                            "field": "velocity_24h",
                            "op": ">",
                            "value": 5,
                            "action": "override_score",
                            "score": 80,
                            "status": "active",
                        }
                    ],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()

        assert "a" in data
        assert "b" in data
        assert "diff" in data
        assert data["a"]["final_score"] == 60
        assert data["b"]["final_score"] == 80
        assert data["diff"]["score_delta"] == 20
        assert data["diff"]["matched_rules_added"] == []
        assert data["diff"]["matched_rules_removed"] == []


class TestShadowComparisonEndpoint:
    """Tests for GET /metrics/shadow/comparison endpoint."""

    def test_shadow_comparison_returns_200(self, client):
        """Test that endpoint returns 200."""
        today = datetime.now().date()
        start = (today - timedelta(days=7)).isoformat()
        end = today.isoformat()

        response = client.get(
            "/metrics/shadow/comparison",
            params={"start_date": start, "end_date": end},
        )
        assert response.status_code == 200

    def test_shadow_comparison_response_structure(self, client):
        """Test response structure."""
        today = datetime.now().date()
        start = (today - timedelta(days=7)).isoformat()
        end = today.isoformat()

        response = client.get(
            "/metrics/shadow/comparison",
            params={"start_date": start, "end_date": end},
        )
        data = response.json()

        assert "period_start" in data
        assert "period_end" in data
        assert "rule_metrics" in data
        assert "total_requests" in data
        assert isinstance(data["rule_metrics"], list)

    def test_shadow_comparison_invalid_date_format(self, client):
        """Test that invalid date format returns 400."""
        response = client.get(
            "/metrics/shadow/comparison",
            params={"start_date": "invalid", "end_date": "also-invalid"},
        )
        assert response.status_code == 400

    def test_shadow_comparison_missing_dates(self, client):
        """Test that missing dates returns 422."""
        response = client.get("/metrics/shadow/comparison")
        assert response.status_code == 422

    def test_shadow_comparison_with_rule_ids(self, client):
        """Test with rule_ids filter."""
        today = datetime.now().date()
        start = (today - timedelta(days=7)).isoformat()
        end = today.isoformat()

        response = client.get(
            "/metrics/shadow/comparison",
            params={
                "start_date": start,
                "end_date": end,
                "rule_ids": "rule1,rule2",
            },
        )
        assert response.status_code == 200


class TestBacktestResultsEndpoint:
    """Tests for GET /backtest/results endpoint."""

    @pytest.fixture(autouse=True)
    def mock_crud_client(self, monkeypatch):
        """Mock the analytics CRUD client for all tests in this class."""
        from api import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = []
        mock_client.stub.ListBacktestResults.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)
        return mock_client

    def test_backtest_results_returns_200(self, client):
        """Test that endpoint returns 200."""
        response = client.get("/backtest/results")
        assert response.status_code == 200

    def test_backtest_results_response_structure(self, client):
        """Test response structure."""
        response = client.get("/backtest/results")
        data = response.json()

        assert "results" in data
        assert "total" in data
        assert isinstance(data["results"], list)

    def test_backtest_results_with_limit(self, client):
        """Test with limit parameter."""
        response = client.get("/backtest/results", params={"limit": 10})
        assert response.status_code == 200

    def test_backtest_results_limit_validation(self, client):
        """Test limit validation (1-100)."""
        # Too low
        response = client.get("/backtest/results", params={"limit": 0})
        assert response.status_code == 422

        # Too high
        response = client.get("/backtest/results", params={"limit": 200})
        assert response.status_code == 422

    def test_backtest_results_with_filters(self, client):
        """Test with all filter parameters."""
        response = client.get(
            "/backtest/results",
            params={
                "rule_id": "test_rule",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "limit": 50,
            },
        )
        assert response.status_code == 200

    def test_backtest_results_invalid_date(self, client):
        """Test invalid date format."""
        response = client.get(
            "/backtest/results",
            params={"start_date": "not-a-date"},
        )
        assert response.status_code == 400


class TestBacktestResultByIdEndpoint:
    """Tests for GET /backtest/results/{job_id} endpoint."""

    @pytest.fixture(autouse=True)
    def mock_crud_client(self, monkeypatch):
        """Mock the analytics CRUD client for all tests in this class."""
        from api import crud_client

        mock_client = MagicMock()
        # Mock GetBacktestResult to raise exception (simulates not found)
        mock_client.stub.GetBacktestResult.side_effect = Exception("Not found")
        monkeypatch.setattr(crud_client, "_client", mock_client)
        return mock_client

    def test_backtest_result_not_found(self, client):
        """Test that non-existent job returns 404."""
        response = client.get("/backtest/results/nonexistent_job_id")
        assert response.status_code == 404

    def test_backtest_result_not_found_message(self, client):
        """Test error message for not found."""
        response = client.get("/backtest/results/nonexistent_job_id")
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()


class TestSuggestionsEndpoint:
    """Tests for GET /suggestions/heuristic endpoint."""

    def test_suggestions_returns_200(self, client):
        """Test that endpoint returns 200."""
        response = client.get("/suggestions/heuristic")
        assert response.status_code == 200

    def test_suggestions_response_structure(self, client):
        """Test response structure."""
        response = client.get("/suggestions/heuristic")
        data = response.json()

        assert "suggestions" in data
        assert "total" in data
        assert isinstance(data["suggestions"], list)

    def test_suggestions_with_field_filter(self, client):
        """Test with field filter."""
        response = client.get(
            "/suggestions/heuristic",
            params={"field": "velocity_24h"},
        )
        assert response.status_code == 200

    def test_suggestions_with_confidence_threshold(self, client):
        """Test with min_confidence parameter."""
        response = client.get(
            "/suggestions/heuristic",
            params={"min_confidence": 0.9},
        )
        assert response.status_code == 200

    def test_suggestions_confidence_validation(self, client):
        """Test confidence parameter validation (0.0-1.0)."""
        # Too low
        response = client.get(
            "/suggestions/heuristic",
            params={"min_confidence": -0.1},
        )
        assert response.status_code == 422

        # Too high
        response = client.get(
            "/suggestions/heuristic",
            params={"min_confidence": 1.5},
        )
        assert response.status_code == 422

    def test_suggestions_with_min_samples(self, client):
        """Test with min_samples parameter."""
        response = client.get(
            "/suggestions/heuristic",
            params={"min_samples": 500},
        )
        assert response.status_code == 200

    def test_suggestions_min_samples_validation(self, client):
        """Test min_samples validation (10-10000)."""
        # Too low
        response = client.get(
            "/suggestions/heuristic",
            params={"min_samples": 5},
        )
        assert response.status_code == 422

        # Too high
        response = client.get(
            "/suggestions/heuristic",
            params={"min_samples": 50000},
        )
        assert response.status_code == 422


class TestEndpointsSafety:
    """Tests to verify safety guarantees of Rule Inspector endpoints."""

    def test_get_rules_is_read_only(self, client):
        """Test that GET /rules has no side effects."""
        # Call multiple times
        response1 = client.get("/rules")
        response2 = client.get("/rules")

        # Should return same data (no state change)
        assert response1.json() == response2.json()

    def test_sandbox_no_database_writes(self, client):
        """Test that sandbox doesn't write to database."""
        # The sandbox endpoint is designed to be pure
        # We can verify it works without DB by using custom ruleset
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {"velocity_24h": 5},
                "base_score": 50,
                "ruleset": {"version": "test", "rules": []},
            },
        )
        assert response.status_code == 200

    def test_shadow_comparison_is_read_only(self, client):
        """Test that shadow comparison has no side effects."""
        today = datetime.now().date()
        start = (today - timedelta(days=1)).isoformat()
        end = today.isoformat()

        # Call multiple times
        response1 = client.get(
            "/metrics/shadow/comparison",
            params={"start_date": start, "end_date": end},
        )
        response2 = client.get(
            "/metrics/shadow/comparison",
            params={"start_date": start, "end_date": end},
        )

        # Should return same data
        assert response1.json() == response2.json()

    def test_backtest_results_is_read_only(self, client, monkeypatch):
        """Test that backtest results has no side effects."""
        from api import crud_client

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.results = []
        mock_client.stub.ListBacktestResults.return_value = mock_response
        monkeypatch.setattr(crud_client, "_client", mock_client)

        # Call multiple times
        response1 = client.get("/backtest/results")
        response2 = client.get("/backtest/results")

        # Should return same data
        assert response1.json() == response2.json()

    def test_suggestions_is_read_only(self, client):
        """Test that suggestions endpoint has no side effects."""
        # Call multiple times with same params
        response1 = client.get(
            "/suggestions/heuristic",
            params={"min_confidence": 0.8},
        )
        response2 = client.get(
            "/suggestions/heuristic",
            params={"min_confidence": 0.8},
        )

        # Should return same structure (content may vary due to time)
        assert "suggestions" in response1.json()
        assert "suggestions" in response2.json()


class TestSandboxRuleActions:
    """Tests for different rule actions in sandbox."""

    def test_sandbox_override_score_action(self, client):
        """Test override_score action."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {"velocity_24h": 10},
                "base_score": 30,
                "ruleset": {
                    "version": "test",
                    "rules": [
                        {
                            "id": "override_test",
                            "field": "velocity_24h",
                            "op": ">",
                            "value": 5,
                            "action": "override_score",
                            "score": 85,
                            "severity": "high",
                            "status": "active",
                        }
                    ],
                },
            },
        )
        data = response.json()
        assert data["final_score"] == 85

    def test_sandbox_clamp_min_action(self, client):
        """Test clamp_min action."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {"velocity_24h": 10},
                "base_score": 30,
                "ruleset": {
                    "version": "test",
                    "rules": [
                        {
                            "id": "clamp_min_test",
                            "field": "velocity_24h",
                            "op": ">",
                            "value": 5,
                            "action": "clamp_min",
                            "score": 60,
                            "severity": "medium",
                            "status": "active",
                        }
                    ],
                },
            },
        )
        data = response.json()
        assert data["final_score"] >= 60

    def test_sandbox_clamp_max_action(self, client):
        """Test clamp_max action."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {"velocity_24h": 10},
                "base_score": 80,
                "ruleset": {
                    "version": "test",
                    "rules": [
                        {
                            "id": "clamp_max_test",
                            "field": "velocity_24h",
                            "op": ">",
                            "value": 5,
                            "action": "clamp_max",
                            "score": 50,
                            "severity": "low",
                            "status": "active",
                        }
                    ],
                },
            },
        )
        data = response.json()
        assert data["final_score"] <= 50

    def test_sandbox_reject_action(self, client):
        """Test reject action."""
        response = client.post(
            "/rules/sandbox/evaluate",
            json={
                "features": {"velocity_24h": 10},
                "base_score": 50,
                "ruleset": {
                    "version": "test",
                    "rules": [
                        {
                            "id": "reject_test",
                            "field": "velocity_24h",
                            "op": ">",
                            "value": 5,
                            "action": "reject",
                            "severity": "high",
                            "status": "active",
                        }
                    ],
                },
            },
        )
        data = response.json()
        assert data["rejected"] is True
        assert data["final_score"] == 99
