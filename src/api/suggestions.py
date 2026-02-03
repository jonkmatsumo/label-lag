"""Heuristic and model-assisted rule suggestion engine using Analytics service."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from google.protobuf.json_format import MessageToDict

from api.crud_client import get_crud_client
from api.model_manager import get_model_manager
from api.rules import Rule, RuleSet, RuleStatus

logger = logging.getLogger(__name__)


@dataclass
class RuleSuggestion:
    """A suggested rule with confidence and evidence."""

    field: str
    operator: str
    threshold: float
    action: str
    suggested_score: int
    confidence: float  # 0.0 to 1.0
    evidence: dict[str, Any]  # Supporting statistics
    reason: str = ""

    def to_rule(self, rule_id: str | None = None) -> Rule:
        """Convert suggestion to a Rule in draft status."""
        if rule_id is None:
            rule_id = f"suggested_{self.field}_{self.operator}_{int(self.threshold)}"

        if self.operator in ["in", "not_in"]:
            value = [int(self.threshold), int(self.threshold) + 1]
        else:
            value = self.threshold

        return Rule(
            id=rule_id,
            field=self.field,
            op=self.operator,
            value=value,
            action=self.action,
            score=self.suggested_score,
            severity="medium",
            reason=self.reason or f"Heuristic suggestion: {self.field} {self.operator} {self.threshold}",
            status=RuleStatus.DRAFT.value,
        )


class SuggestionEngine:
    """Generates rule suggestions from feature distribution analysis via Analytics service."""

    def __init__(self, db_session=None, min_confidence: float = 0.7):
        """Initialize. db_session is ignored."""
        self.min_confidence = min_confidence

    def generate_suggestions(self, field: str | None = None, min_samples: int = 100) -> list[RuleSuggestion]:
        """Generate rule suggestions."""
        suggestions = []
        distributions = self._get_feature_distributions(field, min_samples)

        for field_name, stats in distributions.items():
            field_suggestions = self._analyze_field_distribution(field_name, stats)
            suggestions.extend(field_suggestions)

        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return [s for s in suggestions if s.confidence >= self.min_confidence]

    def _get_feature_distributions(self, field: str | None, min_samples: int) -> dict[str, dict[str, Any]]:
        """Get feature distributions via Analytics service."""
        distributions = {}
        client = get_crud_client()
        
        # Use a large sample for analysis
        resp = client.get_feature_sample(sample_size=2000, stratify=True)
        samples = MessageToDict(resp, preserving_proto_field_name=True).get("samples", [])
        
        if not samples:
            return distributions

        import pandas as pd
        df = pd.DataFrame(samples)
        
        fields_to_analyze = ["velocity_24h", "amount_to_avg_ratio_30d", "balance_volatility_z_score"]
        if field:
            fields_to_analyze = [f for f in fields_to_analyze if f == field]

        for field_name in fields_to_analyze:
            if field_name not in df.columns:
                continue
                
            values = df[field_name].dropna().values
            if len(values) < min_samples:
                continue

            distributions[field_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "percentile_90": float(np.percentile(values, 90)),
                "percentile_95": float(np.percentile(values, 95)),
                "percentile_99": float(np.percentile(values, 99)),
                "percentile_10": float(np.percentile(values, 10)),
                "percentile_5": float(np.percentile(values, 5)),
                "count": len(values),
            }
        return distributions

    def _analyze_field_distribution(self, field_name: str, stats: dict[str, Any]) -> list[RuleSuggestion]:
        """Analyze a field's distribution."""
        suggestions = []
        high_thresholds = [("percentile_90", 0.7), ("percentile_95", 0.8), ("percentile_99", 0.9)]

        for stat_key, confidence in high_thresholds:
            threshold = stats[stat_key]
            if threshold > stats["mean"]:
                suggestions.append(RuleSuggestion(
                    field=field_name, operator=">", threshold=threshold, action="clamp_min",
                    suggested_score=80, confidence=confidence,
                    evidence={"statistic": stat_key, "value": threshold, "mean": stats["mean"], "std": stats["std"], "sample_count": stats["count"]},
                    reason=f"High {field_name} threshold ({stat_key}: {threshold:.2f})"
                ))

        if field_name == "balance_volatility_z_score":
            low_thresholds = [("percentile_10", -2.0, 0.7), ("percentile_5", -2.5, 0.8)]
            for stat_key, default_threshold, confidence in low_thresholds:
                threshold = stats.get(stat_key, default_threshold)
                if threshold < stats["mean"]:
                    suggestions.append(RuleSuggestion(
                        field=field_name, operator="<", threshold=threshold, action="clamp_min",
                        suggested_score=75, confidence=confidence,
                        evidence={"statistic": stat_key, "value": threshold, "mean": stats["mean"], "std": stats["std"], "sample_count": stats["count"]},
                        reason=f"Low balance volatility threshold ({threshold:.2f})"
                    ))
        return suggestions


class ModelAssistedSuggestionEngine:
    """Generates rule suggestions using model insights."""

    def __init__(self, db_session=None, min_importance: float = 0.1):
        """Initialize. db_session is ignored."""
        self.min_importance = min_importance

    def generate_suggestions_from_model(self, sample_size: int = 1000) -> list[RuleSuggestion]:
        """Generate model-assisted suggestions."""
        manager = get_model_manager()
        if not manager.model_loaded:
            return []

        feature_importance = manager.get_feature_importance()
        if not feature_importance:
            return []

        suggestions = []
        for field_name, importance in feature_importance.items():
            if importance < self.min_importance:
                continue
            suggestions.extend(self._analyze_feature_with_model(field_name, importance, sample_size))
        
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions

    def _analyze_feature_with_model(self, field_name: str, importance: float, sample_size: int) -> list[RuleSuggestion]:
        """Analyze a feature using model insights and Analytics service."""
        client = get_crud_client()
        resp = client.get_feature_sample(sample_size=sample_size, stratify=True)
        samples = MessageToDict(resp, preserving_proto_field_name=True).get("samples", [])
        
        if not samples:
            return []

        import pandas as pd
        df = pd.DataFrame(samples)
        if field_name not in df.columns:
            return []
            
        values = df[field_name].dropna().values
        if len(values) < 10:
            return []

        p95 = float(np.percentile(values, 95))
        conf = min(importance * 2.0, 0.95)

        suggestions = []
        if p95 > np.mean(values):
            suggestions.append(RuleSuggestion(
                field=field_name, operator=">", threshold=p95, action="clamp_min",
                suggested_score=85, confidence=conf,
                evidence={"feature_importance": importance, "percentile_95": p95, "mean": float(np.mean(values)), "sample_count": len(values)},
                reason=f"High-importance ({importance:.3f}): {field_name} > {p95:.2f}"
            ))
        return suggestions


_global_suggestion_engine = None

def get_suggestion_engine() -> SuggestionEngine:
    global _global_suggestion_engine
    if _global_suggestion_engine is None:
        _global_suggestion_engine = SuggestionEngine()
    return _global_suggestion_engine