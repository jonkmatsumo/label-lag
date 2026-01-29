import os

import pytest
import requests

FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://localhost:8100")
GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "http://localhost:8181")

FIXTURES = [
    {
        "user_id": "user_001",
        "amount": 100.0,
        "currency": "USD",
        "client_transaction_id": "txn_001",
    },
    {
        "user_id": "user_999",
        "amount": 2500.5,
        "currency": "USD",
        "client_transaction_id": "txn_002",
    },
]


def _ensure_service(url: str) -> None:
    response = requests.get(f"{url}/health", timeout=5)
    response.raise_for_status()


def _call_evaluate(url: str, payload: dict) -> dict:
    response = requests.post(f"{url}/evaluate/signal", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def _normalize_response(payload: dict) -> dict:
    normalized = dict(payload)
    normalized.pop("request_id", None)

    def _sort_list(items, keys):
        def sort_key(item):
            return tuple(item.get(k, "") for k in keys)

        return sorted(items or [], key=sort_key)

    normalized["risk_components"] = _sort_list(
        normalized.get("risk_components"), ["key", "label"]
    )
    normalized["matched_rules"] = _sort_list(
        normalized.get("matched_rules"), ["rule_id", "severity", "reason"]
    )
    normalized["shadow_matched_rules"] = _sort_list(
        normalized.get("shadow_matched_rules"), ["rule_id", "severity", "reason"]
    )

    return normalized


@pytest.fixture(scope="module", autouse=True)
def _check_services():
    try:
        _ensure_service(FASTAPI_BASE_URL)
        _ensure_service(GATEWAY_BASE_URL)
    except requests.RequestException as exc:
        pytest.skip(f"Services not reachable: {exc}")


@pytest.mark.parametrize("payload", FIXTURES)
def test_gateway_matches_fastapi_for_fixture(payload):
    fastapi_resp = _normalize_response(_call_evaluate(FASTAPI_BASE_URL, payload))
    gateway_resp = _normalize_response(_call_evaluate(GATEWAY_BASE_URL, payload))

    assert gateway_resp == fastapi_resp


def test_gateway_matches_rule_outputs_when_present():
    payload = FIXTURES[0]
    fastapi_resp = _call_evaluate(FASTAPI_BASE_URL, payload)

    if not fastapi_resp.get("matched_rules") and not fastapi_resp.get(
        "shadow_matched_rules"
    ):
        pytest.skip("FastAPI returned no rule matches for configured ruleset.")

    gateway_resp = _normalize_response(_call_evaluate(GATEWAY_BASE_URL, payload))
    fastapi_resp = _normalize_response(fastapi_resp)

    assert gateway_resp["matched_rules"] == fastapi_resp["matched_rules"]
    assert gateway_resp["shadow_matched_rules"] == fastapi_resp["shadow_matched_rules"]
