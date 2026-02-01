"""Parity test harness for comparing Python API and Go Gateway."""

import os
from typing import Any

import pytest
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8100")
GATEWAY_BASE_URL = os.getenv("GATEWAY_BASE_URL", "http://localhost:8081")
RUN_PARITY_TESTS = os.getenv("RUN_PARITY_TESTS", "false").lower() == "true"

class ParityHarness:
    """Helper to compare responses between API and Gateway."""

    def __init__(self, api_base: str, gateway_base: str):
        self.api_base = api_base
        self.gateway_base = gateway_base

    def compare_inference(self, payload: dict[str, Any], tolerance: float = 1e-5):
        """Compare inference results from both services."""
        api_resp = requests.post(f"{self.api_base}/evaluate/signal", json=payload)
        api_resp.raise_for_status()
        api_data = api_resp.json()

        # Gateway uses same path as API
        gateway_resp = requests.post(f"{self.gateway_base}/evaluate/signal", json=payload)
        gateway_resp.raise_for_status()
        gateway_data = gateway_resp.json()

        self._assert_parity(api_data, gateway_data, tolerance)

    def _assert_parity(self, api: dict[str, Any], gateway: dict[str, Any], tolerance: float):
        """Assert semantic parity with tolerance and normalization."""
        # Compare core fields
        assert abs(api["score"] - gateway["score"]) <= tolerance

        # Normalize and compare matched rules
        api_rules = sorted(api.get("matched_rules", []), key=lambda x: x["rule_id"])
        gateway_rules = sorted(gateway.get("matched_rules", []), key=lambda x: x["rule_id"])

        assert len(api_rules) == len(gateway_rules)
        for a, g in zip(api_rules, gateway_rules):
            assert a["rule_id"] == g["rule_id"]
            if "score_adjustment" in a:
                assert abs(a["score_adjustment"] - g["score_adjustment"]) <= tolerance

@pytest.mark.skipif(not RUN_PARITY_TESTS, reason="Parity tests disabled")
def test_inference_parity_basic():
    """Basic inference parity check."""
    harness = ParityHarness(API_BASE_URL, GATEWAY_BASE_URL)
    payload = {
        "user_id": "parity_user_1",
        "amount": 100.0,
        "currency": "USD",
        "client_transaction_id": "parity_txn_1"
    }
    harness.compare_inference(payload)
