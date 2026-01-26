"""Attribution of model score changes to rules."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from api.inference_log import InferenceLogger

logger = logging.getLogger(__name__)


@dataclass
class RuleAttribution:
    """Attribution metrics for a single rule."""
    
    rule_id: str
    total_matches: int
    mean_model_score: float  # Avg score BEFORE rule
    mean_final_score: float  # Avg score AFTER rule
    mean_impact: float       # Avg absolute delta
    
    @property
    def net_impact(self) -> float:
        """Net impact (final - model)."""
        return self.mean_final_score - self.mean_model_score
        
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "total_matches": self.total_matches,
            "mean_model_score": self.mean_model_score,
            "mean_final_score": self.mean_final_score,
            "mean_impact": self.mean_impact,
            "net_impact": self.net_impact
        }


class AttributionService:
    """Service to compute rule attribution metrics."""
    
    def __init__(self, logger: InferenceLogger | None = None):
        """Initialize service."""
        self.logger = logger or InferenceLogger()
        
    def get_rule_attribution(
        self, 
        rule_id: str, 
        start_time: datetime,
        end_time: datetime
    ) -> RuleAttribution | None:
        """Compute attribution metrics for a specific rule."""
        
        # Query events involving this rule
        events = self.logger.query_events(start_time, end_time, rule_id)
        
        if not events:
            return None
            
        total_matches = 0
        total_model_score = 0
        total_final_score = 0
        total_impact = 0.0
        
        for event in events:
            # Check if rule was actually applied (is_shadow=False)
            impacts = event.get("rule_impacts", [])
            
            # Find this rule's impact
            rule_impact = next((r for r in impacts if r["rule_id"] == rule_id), None)
            
            if not rule_impact:
                continue
                
            # We track stats for both production and shadow matches differently
            # But the requirement asks "Why did the model score change?"
            # So we focus on production application usually.
            # However, for analytics, shadow impact is also useful ("What WOULD it have done?").
            # But InferenceEvent.final_score is the ACTUAL final score.
            # If this rule was shadow, it didn't affect final_score.
            # We should rely on the impact recorded in the event.
            
            is_shadow = rule_impact.get("is_shadow", False)
            delta = rule_impact.get("score_delta", 0.0)
            
            total_matches += 1
            total_model_score += event["model_score"]
            
            # If shadow, the "final" score from THIS rule's perspective would be model + delta
            # If production, it contributed to the actual final_score.
            # To isolate this rule's effect, we can say:
            # impact = delta
            # associated_model_score = event.model_score
            # associated_result_score = event.model_score + delta (Theoretical effect)
            
            # Use theoretical effect for cleaner attribution of single rule
            total_final_score += (event["model_score"] + delta)
            total_impact += abs(delta)
            
        if total_matches == 0:
            return None
            
        return RuleAttribution(
            rule_id=rule_id,
            total_matches=total_matches,
            mean_model_score=total_model_score / total_matches,
            mean_final_score=total_final_score / total_matches,
            mean_impact=total_impact / total_matches
        )
