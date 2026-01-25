"""Heuristic rule suggestion engine based on feature distributions."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import text

from api.rules import Rule, RuleSet, RuleStatus
from synthetic_pipeline.db.session import DatabaseSession

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
        """Convert suggestion to a Rule in draft status.

        Args:
            rule_id: Optional rule ID. If None, generates one.

        Returns:
            Rule instance in draft status.
        """
        if rule_id is None:
            rule_id = f"suggested_{self.field}_{self.operator}_{int(self.threshold)}"

        # Determine value type based on operator
        if self.operator in ["in", "not_in"]:
            # For list operators, create a list around the threshold
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
    """Generates rule suggestions from feature distribution analysis."""

    def __init__(
        self,
        db_session: DatabaseSession | None = None,
        min_confidence: float = 0.7,
    ):
        """Initialize suggestion engine.

        Args:
            db_session: Database session. If None, creates a new one.
            min_confidence: Minimum confidence threshold for suggestions.
        """
        self.db_session = db_session or DatabaseSession()
        self.min_confidence = min_confidence

    def generate_suggestions(
        self,
        field: str | None = None,
        min_samples: int = 100,
    ) -> list[RuleSuggestion]:
        """Generate rule suggestions from feature distributions.

        Args:
            field: Optional field to analyze. If None, analyzes all fields.
            min_samples: Minimum number of samples required for analysis.

        Returns:
            List of RuleSuggestion objects.
        """
        suggestions = []

        # Get feature distributions
        distributions = self._get_feature_distributions(field, min_samples)

        for field_name, stats in distributions.items():
            field_suggestions = self._analyze_field_distribution(field_name, stats)
            suggestions.extend(field_suggestions)

        # Sort by confidence (highest first)
        suggestions.sort(key=lambda s: s.confidence, reverse=True)

        # Filter by minimum confidence
        suggestions = [s for s in suggestions if s.confidence >= self.min_confidence]

        logger.info(f"Generated {len(suggestions)} rule suggestions")

        return suggestions

    def _get_feature_distributions(
        self, field: str | None, min_samples: int
    ) -> dict[str, dict[str, Any]]:
        """Get feature distributions from database.

        Args:
            field: Optional field to analyze.
            min_samples: Minimum samples required.

        Returns:
            Dictionary mapping field names to distribution statistics.
        """
        distributions = {}

        with self.db_session.get_session() as session:
            # Get all numeric feature fields
            fields_to_analyze = [
                "velocity_24h",
                "amount_to_avg_ratio_30d",
                "balance_volatility_z_score",
            ]

            if field is not None:
                fields_to_analyze = [f for f in fields_to_analyze if f == field]

            for field_name in fields_to_analyze:
                query = text(f"""
                    SELECT {field_name}
                    FROM feature_snapshots
                    WHERE {field_name} IS NOT NULL
                """)

                result = session.execute(query)
                values = [float(row[0]) for row in result]

                if len(values) < min_samples:
                    continue

                # Compute statistics
                values_array = np.array(values)
                distributions[field_name] = {
                    "values": values,
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "min": float(np.min(values_array)),
                    "max": float(np.max(values_array)),
                    "percentile_75": float(np.percentile(values_array, 75)),
                    "percentile_90": float(np.percentile(values_array, 90)),
                    "percentile_95": float(np.percentile(values_array, 95)),
                    "percentile_99": float(np.percentile(values_array, 99)),
                    "count": len(values),
                }

        return distributions

    def _analyze_field_distribution(
        self, field_name: str, stats: dict[str, Any]
    ) -> list[RuleSuggestion]:
        """Analyze a field's distribution and generate suggestions.

        Args:
            field_name: Name of the field.
            stats: Distribution statistics.

        Returns:
            List of RuleSuggestion objects.
        """
        suggestions = []

        # Suggest high-value thresholds (for clamp_min or reject)
        high_thresholds = [
            ("percentile_90", 0.7),
            ("percentile_95", 0.8),
            ("percentile_99", 0.9),
        ]

        for stat_key, confidence in high_thresholds:
            threshold = stats[stat_key]
            if threshold > stats["mean"]:
                # High threshold suggests risk - use clamp_min or reject
                suggestions.append(
                    RuleSuggestion(
                        field=field_name,
                        operator=">",
                        threshold=threshold,
                        action="clamp_min",
                        suggested_score=80,
                        confidence=confidence,
                        evidence={
                            "statistic": stat_key,
                            "value": threshold,
                            "mean": stats["mean"],
                            "std": stats["std"],
                            "sample_count": stats["count"],
                        },
                        reason=f"High {field_name} threshold ({stat_key}: {threshold:.2f})",
                    )
                )

        # Suggest low-value thresholds for balance volatility (negative z-scores)
        if field_name == "balance_volatility_z_score":
            low_thresholds = [
                ("percentile_10", -2.0, 0.7),
                ("percentile_5", -2.5, 0.8),
            ]

            for stat_key, default_threshold, confidence in low_thresholds:
                # For negative z-scores, use percentile if available, else default
                if stat_key in stats:
                    threshold = stats[stat_key]
                else:
                    threshold = default_threshold

                if threshold < stats["mean"]:
                    suggestions.append(
                        RuleSuggestion(
                            field=field_name,
                            operator="<",
                            threshold=threshold,
                            action="clamp_min",
                            suggested_score=75,
                            confidence=confidence,
                            evidence={
                                "statistic": stat_key,
                                "value": threshold,
                                "mean": stats["mean"],
                                "std": stats["std"],
                                "sample_count": stats["count"],
                            },
                            reason=f"Low balance volatility threshold ({threshold:.2f})",
                        )
                    )

        return suggestions

    def create_ruleset_from_suggestions(
        self, suggestions: list[RuleSuggestion], version: str = "suggested_v1"
    ) -> RuleSet:
        """Create a RuleSet from suggestions.

        Args:
            suggestions: List of RuleSuggestion objects.
            version: Version string for the ruleset.

        Returns:
            RuleSet with suggested rules in draft status.
        """
        rules = [s.to_rule() for s in suggestions]
        return RuleSet(version=version, rules=rules)


# Global suggestion engine instance
_global_suggestion_engine: SuggestionEngine | None = None


def get_suggestion_engine() -> SuggestionEngine:
    """Get the global suggestion engine instance.

    Returns:
        Global SuggestionEngine instance.
    """
    global _global_suggestion_engine
    if _global_suggestion_engine is None:
        _global_suggestion_engine = SuggestionEngine()
    return _global_suggestion_engine


def set_suggestion_engine(engine: SuggestionEngine) -> None:
    """Set the global suggestion engine instance (for testing).

    Args:
        engine: SuggestionEngine instance to use.
    """
    global _global_suggestion_engine
    _global_suggestion_engine = engine
