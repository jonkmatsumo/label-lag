import os

import pytest

from training.gateway_client import get_gateway_client, reset_gateway_client
from training.gateway_grpc_client import GatewayGrpcClient


# Skip integration tests if not explicitly enabled or if services aren't running
@pytest.mark.skipif(
    os.getenv("TRAINING_GATEWAY_TRANSPORT") != "grpc",
    reason="Skipping gRPC integration tests: TRAINING_GATEWAY_TRANSPORT != grpc",
)
class TestGatewayIntegration:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        # Ensure we are using gRPC transport
        os.environ["TRAINING_GATEWAY_TRANSPORT"] = "grpc"
        # Ensure address is set (defaulting to localhost for local testing)
        if "INFERENCE_GATEWAY_GRPC_ADDR" not in os.environ:
            os.environ["INFERENCE_GATEWAY_GRPC_ADDR"] = "localhost:50505"

        reset_gateway_client()
        yield
        reset_gateway_client()

    def test_client_is_grpc(self):
        client = get_gateway_client()
        assert isinstance(client, GatewayGrpcClient)

    def test_evaluate_rules(self):
        client = get_gateway_client()
        features = {"velocity_24h": 100}
        base_score = 50

        result = client.evaluate_rules(features, base_score)

        assert "final_score" in result
        # Default rules shouldn't change score for 100 velocity
        assert result["final_score"] == 50
        assert "evaluation_time_ms" in result

    def test_evaluate_rules_shadow_mode(self):
        client = get_gateway_client()
        features = {"velocity_24h": 1000}  # High velocity might trigger a rule
        base_score = 50

        # Pass a custom ruleset to trigger a rule
        ruleset = {
            "rules": [
                {
                    "id": "test_rule",
                    "field": "velocity_24h",
                    "op": ">",
                    "value": 500,
                    "action": "reject",
                    "score": 0,
                    "severity": "high",
                }
            ]
        }

        # Shadow mode = True
        result = client.evaluate_rules(
            features, base_score, ruleset=ruleset, shadow_mode=True
        )

        # In shadow mode, final_score should NOT be affected by the rule
        assert result["final_score"] == 50
        assert result["shadow_score"] == 99  # Reject action sets score to 99
        assert "test_rule" in result["matched_rules"]
        assert not result["rejected"]  # Main result not rejected

    def test_diff_rules(self):
        client = get_gateway_client()
        features = {"velocity_24h": 100}
        base_score = 50

        ruleset_a = {"rules": []}
        ruleset_b = {
            "rules": [
                {
                    "id": "new_rule",
                    "field": "velocity_24h",
                    "op": ">",
                    "value": 10,
                    "action": "reject",
                    "severity": "high",
                }
            ]
        }

        result = client.diff_rules(
            features, base_score, ruleset_a=ruleset_a, ruleset_b=ruleset_b
        )

        assert "diff" in result
        diff = result["diff"]
        assert diff["severity"] != "none"  # Should have severity
        assert "new_rule" in diff["matched_rules_added"]
