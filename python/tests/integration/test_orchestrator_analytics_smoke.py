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


def _get_without_tenant(
    path: str, params: dict[str, object] | None = None
) -> requests.Response:
    return requests.get(
        f"{ORCHESTRATOR_BASE_URL}{path}",
        params=params or {},
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


def _parse_event_sort_key(event: dict) -> tuple[datetime, int] | None:
    timestamp_value: datetime | None = None
    for key in ("timestamp", "created_at", "createdAt", "ts"):
        raw = event.get(key)
        if isinstance(raw, str) and raw:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            timestamp_value = parsed.astimezone(UTC)
            break
    if timestamp_value is None:
        return None

    event_id = 0
    for key in ("event_id", "eventId", "id"):
        raw = event.get(key)
        if isinstance(raw, int):
            event_id = raw
            break
        if isinstance(raw, str) and raw.isdigit():
            event_id = int(raw)
            break

    return (timestamp_value, event_id)


def _assert_event_order_non_increasing(events: list[dict]) -> None:
    sort_keys = [key for key in (_parse_event_sort_key(item) for item in events) if key]
    for idx in range(1, len(sort_keys)):
        previous = sort_keys[idx - 1]
        current = sort_keys[idx]
        assert previous >= current, (
            f"job events are not in stable descending order at index {idx}: "
            f"{previous} then {current}"
        )


@pytest.fixture(scope="module")
def orchestrator_ready() -> str:
    try:
        resp = _get("/health")
        resp.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"orchestrator not reachable at {ORCHESTRATOR_BASE_URL}: {exc}")
    return ORCHESTRATOR_BASE_URL


def test_jobs_tenant_header_required(orchestrator_ready: str) -> None:
    resp = _get_without_tenant("/jobs", {"limit": 1, "offset": 0})
    assert resp.status_code == 400, resp.text
    payload = _as_object(resp)
    detail = payload.get("detail")
    assert isinstance(detail, str) and detail.strip()


def test_job_events_cursor_pagination_smoke(orchestrator_ready: str) -> None:
    jobs_resp = _get("/jobs", {"limit": 5, "offset": 0})
    assert jobs_resp.status_code == 200, jobs_resp.text
    jobs_payload = _as_object(jobs_resp)
    jobs = jobs_payload.get("jobs") or jobs_payload.get("items") or []
    assert isinstance(jobs, list)
    if not jobs:
        return

    job_id = _first_id(jobs, "jobId", "job_id", "id")
    assert job_id, "non-empty /jobs response did not include a job id"

    events_resp = _get(f"/jobs/{job_id}/events", {"limit": 5, "offset": 0})
    assert events_resp.status_code == 200, events_resp.text
    events_payload = _as_object(events_resp)
    event_items = events_payload.get("events") or events_payload.get("items") or []
    assert isinstance(event_items, list)
    _assert_event_order_non_increasing(
        [item for item in event_items if isinstance(item, dict)]
    )

    next_cursor = (
        events_payload.get("next_cursor")
        or events_payload.get("nextCursor")
        or events_payload.get("cursor")
    )
    before_ts = (
        events_payload.get("next_before_ts")
        or events_payload.get("before_ts")
        or events_payload.get("nextBeforeTs")
        or events_payload.get("beforeTs")
    )
    before_id = (
        events_payload.get("next_before_id")
        or events_payload.get("before_id")
        or events_payload.get("nextBeforeId")
        or events_payload.get("beforeId")
    )

    if event_items and isinstance(next_cursor, str) and next_cursor:
        cursor_resp = _get(
            f"/jobs/{job_id}/events", {"limit": 5, "cursor": next_cursor}
        )
        assert cursor_resp.status_code == 200, cursor_resp.text
        cursor_payload = _as_object(cursor_resp)
        cursor_items = cursor_payload.get("events") or cursor_payload.get("items") or []
        assert isinstance(cursor_items, list)
        _assert_event_order_non_increasing(
            [item for item in cursor_items if isinstance(item, dict)]
        )
        return

    if event_items and isinstance(before_ts, str) and before_ts:
        cursor_params: dict[str, object] = {"limit": 5, "before_ts": before_ts}
        if isinstance(before_id, int):
            cursor_params["before_id"] = before_id
        elif isinstance(before_id, str) and before_id.isdigit():
            cursor_params["before_id"] = int(before_id)

        cursor_resp = _get(f"/jobs/{job_id}/events", cursor_params)
        assert cursor_resp.status_code == 200, cursor_resp.text
        cursor_payload = _as_object(cursor_resp)
        cursor_items = cursor_payload.get("events") or cursor_payload.get("items") or []
        assert isinstance(cursor_items, list)
        _assert_event_order_non_increasing(
            [item for item in cursor_items if isinstance(item, dict)]
        )
        return

    offset_resp = _get(f"/jobs/{job_id}/events", {"limit": 5, "offset": 5})
    assert offset_resp.status_code == 200, offset_resp.text
    offset_payload = _as_object(offset_resp)
    offset_items = offset_payload.get("events") or offset_payload.get("items") or []
    assert isinstance(offset_items, list)


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
