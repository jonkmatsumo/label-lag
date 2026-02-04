import os
import requests
import uuid
import logging
from typing import Any

logger = logging.getLogger(__name__)

class GatewayDecisionClient:
    """Client for calling the Go Inference Gateway for rule-based decisioning."""

    def __init__(self, base_url: str = None, timeout: float = 5.0):
        # Allow override via parameter or environment variable
        self.base_url = base_url or os.getenv("INFERENCE_GATEWAY_URL", "http://inference-gateway:8081")
        # Ensure no trailing slash
        self.base_url = self.base_url.rstrip("/")
        self.timeout = timeout

    def evaluate_rules(
        self, 
        features: dict[str, Any], 
        base_score: int, 
        ruleset: dict[str, Any] = None, 
        request_id: str = None
    ) -> dict[str, Any]:
        """
        Evaluate rules against a set of features using the Go gateway.
        
        Args:
            features: Dictionary of feature names and values.
            base_score: Model score (1-99) to adjust.
            ruleset: Optional custom RuleSet definition. If None, gateway uses production ruleset.
            request_id: Optional request identifier for tracing.
            
        Returns:
            Dictionary containing final_score, matched_rules, explanations, etc.
            
        Raises:
            RuntimeError: If the gateway call fails or returns an error status.
        """
        if request_id is None:
            request_id = f"req_{uuid.uuid4().hex[:12]}"
        
        url = f"{self.base_url}/evaluate/rules"
        payload = {
            "features": features,
            "base_score": base_score,
        }
        if ruleset is not None:
            payload["ruleset"] = ruleset
            
        headers = {
            "X-Request-Id": request_id,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    if "detail" in error_json:
                        error_detail = error_json["detail"]
                except Exception:
                    pass
                
                logger.error(f"Gateway rule evaluation failed ({response.status_code}): {error_detail}")
                if response.status_code == 400:
                    raise ValueError(f"Gateway evaluation failed (400): {error_detail}")
                raise RuntimeError(f"Gateway evaluation failed ({response.status_code}): {error_detail}")
                
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Gateway rule evaluation timed out after {self.timeout}s")
            raise RuntimeError(f"Gateway evaluation timed out after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            logger.error(f"Gateway rule evaluation request failed: {e}")
            raise RuntimeError(f"Gateway evaluation request failed: {str(e)}")

_client = None

def get_gateway_client() -> GatewayDecisionClient:
    """Get the singleton GatewayDecisionClient instance."""
    global _client
    if _client is None:
        _client = GatewayDecisionClient()
    return _client

def reset_gateway_client():
    """Reset the singleton client (useful for testing)."""
    global _client
    _client = None
