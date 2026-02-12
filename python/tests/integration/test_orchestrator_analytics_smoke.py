"""Smoke coverage for orchestrator analytics endpoints.

These tests exercise Jobs / Training / Profiles read paths through the
orchestrator HTTP API using tenant headers.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
import requests

ORCHESTRATOR_BASE_URL = os.getenv(
    "ORCHESTRATOR_BASE_URL", "http://localhost:8081"
).rstrip("/")
TENANT_ID = os.getenv("TENANT_ID", "tenant-1")
TIMEOUT = float(os.getenv("SMOKE_TIMEOUT_SECONDS", "10"))


def _headers() -> dict[str, str]:
    return {"X-Tenant-Id": TENANT_ID}


def _get(path: str, params: dict[str, object] | None = None) -> requests.Response:
    return requests.get(
        f"{ORCHESTRATOR_BASE_URL}{path}",
        params=params or {},
        headers=_headers(),
        timeout=TIMEOUT,
    )


def _as_object(resp: requests.Response) -> dict:
    payload = resp.json()
    assert isinstance(payload, dict), f"expected JSON object, got {type(payload)}"
    return payload


def _first_id(items: list[dict], *keys: str) -> str | None:
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
    return None


@pytest.fixture(scope="module")
def orchestrator_ready() -> str:
    try:
        resp = _get("/health")
        resp.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"orchestrator not reachable at {ORCHESTRATOR_BASE_URL}: {exc}")
    return ORCHESTRATOR_BASE_URL


def test_jobs_endpoints_smoke(orchestrator_ready: str) -> None:
    resp = _get("/jobs", {"limit": 5, "offset": 0})
    assert resp.status_code == 200, resp.text
    payload = _as_object(resp)

    jobs = payload.get("jobs") or payload.get("items") or []
    assert isinstance(jobs, list)
    if not jobs:
        return

    job_id = _first_id(jobs, "jobId", "job_id", "id")
    assert job_id, "non-empty /jobs response did not include a job id"

    detail = _get(f"/jobs/{job_id}")
    assert detail.status_code == 200, detail.text

    events = _get(f"/jobs/{job_id}/events", {"limit": 10, "offset": 0})
    assert events.status_code == 200, events.text
    events_payload = _as_object(events)
    event_items = events_payload.get("events") or events_payload.get("items") or []
    assert isinstance(event_items, list)


def test_training_endpoints_smoke(orchestrator_ready: str) -> None:
    resp = _get("/training-runs", {"limit": 5, "offset": 0})
    assert resp.status_code == 200, resp.text
    payload = _as_object(resp)

    runs = (
        payload.get("runs") or payload.get("trainingRuns") or payload.get("items") or []
    )
    assert isinstance(runs, list)

    model_name = "default-model"
    if runs:
        run_id = _first_id(runs, "runId", "run_id", "id")
        assert run_id, "non-empty /training-runs response did not include a run id"

        detail = _get(f"/training-runs/{run_id}")
        assert detail.status_code == 200, detail.text

        first = runs[0] if isinstance(runs[0], dict) else {}
        model_name = first.get("modelName") or first.get("model_name") or model_name

    now = datetime.now(UTC)
    start = (now - timedelta(days=30)).isoformat().replace("+00:00", "Z")
    end = now.isoformat().replace("+00:00", "Z")
    series = _get(
        "/metrics/series",
        {
            "model_name": model_name,
            "metric_name": "accuracy",
            "start_date": start,
            "end_date": end,
        },
    )
    assert series.status_code in (200, 400), series.text


def test_profiles_endpoints_smoke(orchestrator_ready: str) -> None:
    resp = _get("/dataset/profiles", {"limit": 5, "offset": 0})
    assert resp.status_code == 200, resp.text
    payload = _as_object(resp)

    profiles = payload.get("profiles") or payload.get("items") or []
    assert isinstance(profiles, list)
    if not profiles:
        summary = _get("/dataset/summary", {"profile_id": "latest"})
        assert summary.status_code in (200, 400, 404), summary.text
        return

    profile_id = _first_id(profiles, "profileId", "profile_id", "id")
    assert profile_id, (
        "non-empty /dataset/profiles response did not include a profile id"
    )

    detail = _get(f"/dataset/profiles/{profile_id}")
    assert detail.status_code in (200, 404), detail.text

    summary = _get("/dataset/summary", {"profile_id": profile_id})
    assert summary.status_code in (200, 400, 404), summary.text
