"""Regression guardrails for stabilization branch.

These tests ensure that:
1. No mandatory auth is enforced by default (JWT, RBAC)
2. No async job conversion has happened without explicit opt-in
3. All feature flags default to legacy/existing behavior

Per BRANCH_PROTOCOL.md, these guardrails protect against regression.
"""

from fastapi.testclient import TestClient

from api.draft_store import get_draft_store, reset_draft_store
from api.inference_backend_config import get_inference_backend
from api.inference_event_sink import (
    get_inference_event_sink,
    reset_inference_event_sink,
)
from api.main import app
from api.rule_store import get_rule_store_backend
from pipeline.materialize_features import get_materialization_mode


class TestNoMandatoryAuth:
    """Verify endpoints work without authentication headers.

    Per BRANCH_PROTOCOL.md: NO mandatory JWT or RBAC enforcement.
    Endpoints must work without Authorization headers.
    """

    def setup_method(self):
        """Reset state before each test."""
        reset_draft_store()
        reset_inference_event_sink()
        self.client = TestClient(app)

    def test_health_no_auth_required(self):
        """Health check works without auth headers."""
        response = self.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_evaluate_signal_no_auth_required(self):
        """Signal evaluation works without auth headers."""
        response = self.client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_test_123",
                "amount": 100.00,
                "currency": "USD",
                "client_transaction_id": "txn_guardrail_test",
            },
        )
        # Should succeed without auth (200 for successful evaluation)
        assert response.status_code == 200
        data = response.json()
        # Verify synchronous response (not async job reference)
        assert "score" in data
        assert "request_id" in data

    def test_stats_endpoint_no_auth_required(self):
        """Stats endpoint works without auth headers."""
        response = self.client.get("/stats")
        # May return 200 or 503 depending on DB, but not 401/403
        assert response.status_code not in (401, 403)

    def test_no_www_authenticate_header(self):
        """Responses do not include WWW-Authenticate header (no auth challenge)."""
        response = self.client.get("/health")
        assert "WWW-Authenticate" not in response.headers


class TestSynchronousEndpoints:
    """Verify endpoints return synchronous results, not async job references.

    Per BRANCH_PROTOCOL.md: NO async job conversion by default.
    """

    def setup_method(self):
        """Reset state before each test."""
        reset_draft_store()
        reset_inference_event_sink()
        self.client = TestClient(app)

    def test_evaluate_signal_returns_score_not_job_id(self):
        """Signal evaluation returns score directly, not async job reference."""
        response = self.client.post(
            "/evaluate/signal",
            json={
                "user_id": "user_sync_test",
                "amount": 250.00,
                "currency": "USD",
                "client_transaction_id": "txn_sync_guardrail",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Must have immediate score, not pending job
        assert "score" in data
        assert isinstance(data["score"], int)
        assert 1 <= data["score"] <= 99

        # Must NOT be async job response pattern
        assert "job_id" not in data or data.get("status") != "pending"
        assert "poll_url" not in data

    def test_health_check_is_synchronous(self):
        """Health check returns immediately, not as background task."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()

        # Immediate status, not async
        assert data["status"] == "healthy"
        assert "job_id" not in data


class TestFeatureFlagsDefaultToLegacy:
    """Verify all feature flags default to legacy/existing behavior.

    Per BRANCH_PROTOCOL.md: All flags must default to legacy.
    """

    def test_rule_store_backend_defaults_to_inmemory(self, monkeypatch):
        """RULE_STORE_BACKEND defaults to 'inmemory', not postgres."""
        monkeypatch.delenv("RULE_STORE_BACKEND", raising=False)
        assert get_rule_store_backend() == "inmemory"

    def test_inference_event_sink_defaults_to_jsonl(self, monkeypatch):
        """INFERENCE_EVENT_SINK defaults to 'jsonl', not postgres or none."""
        monkeypatch.delenv("INFERENCE_EVENT_SINK", raising=False)
        reset_inference_event_sink()
        from api.inference_event_sink import JsonlFileSink

        sink = get_inference_event_sink()
        assert isinstance(sink, JsonlFileSink)

    def test_materialization_mode_defaults_to_legacy(self, monkeypatch):
        """FEATURE_MATERIALIZATION_MODE defaults to 'legacy', not cursor."""
        monkeypatch.delenv("FEATURE_MATERIALIZATION_MODE", raising=False)
        assert get_materialization_mode() == "legacy"

    def test_inference_backend_defaults_to_python(self, monkeypatch):
        """INFERENCE_BACKEND defaults to 'python', not go."""
        monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
        assert get_inference_backend() == "python"


class TestNoUnexpectedDependencies:
    """Verify no unexpected database or service dependencies in defaults."""

    def test_draft_store_does_not_require_postgres_by_default(self, monkeypatch):
        """DraftRuleStore works without postgres when using default inmemory backend."""
        monkeypatch.delenv("RULE_STORE_BACKEND", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        reset_draft_store()

        # Should not raise - inmemory doesn't need DB
        store = get_draft_store()
        assert store is not None

    def test_inference_sink_does_not_require_postgres_by_default(self, monkeypatch):
        """InferenceEventSink works without postgres (default jsonl backend)."""
        monkeypatch.delenv("INFERENCE_EVENT_SINK", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        reset_inference_event_sink()

        # Should not raise - jsonl doesn't need DB
        sink = get_inference_event_sink()
        assert sink is not None
