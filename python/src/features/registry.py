"""Canonical feature registry for the fraud detection platform."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class FeatureDefinition:
    """Definition of a single feature."""

    name: str
    dtype: str  # "float", "int", "str", "bool"
    group: str  # "transaction", "user", "merchant", "network"
    description: str


class FeatureRegistry:
    """Registry of all available features."""

    _features: ClassVar[dict[str, FeatureDefinition]] = {
        # Core model features
        "velocity_24h": FeatureDefinition(
            name="velocity_24h",
            dtype="int",
            group="transaction",
            description="Number of transactions by user in last 24 hours",
        ),
        "amount_to_avg_ratio_30d": FeatureDefinition(
            name="amount_to_avg_ratio_30d",
            dtype="float",
            group="transaction",
            description="Ratio of current amount to 30-day average amount",
        ),
        "balance_volatility_z_score": FeatureDefinition(
            name="balance_volatility_z_score",
            dtype="float",
            group="user",
            description="Z-score of balance volatility",
        ),
        # Additional available features (inference context)
        "amount": FeatureDefinition(
            name="amount",
            dtype="float",
            group="transaction",
            description="Transaction amount",
        ),
        "merchant_risk_score": FeatureDefinition(
            name="merchant_risk_score",
            dtype="int",
            group="merchant",
            description="Risk score of the merchant",
        ),
        "is_off_hours_txn": FeatureDefinition(
            name="is_off_hours_txn",
            dtype="bool",
            group="transaction",
            description="Whether transaction occurred during off-hours",
        ),
    }

    @classmethod
    def get(cls, name: str) -> FeatureDefinition:
        """Get feature definition by name."""
        if name not in cls._features:
            raise ValueError(f"Unknown feature: {name}")
        return cls._features[name]

    @classmethod
    def list_features(cls) -> list[str]:
        """List all registered feature names."""
        return list(cls._features.keys())

    @classmethod
    def expand_group(cls, group: str) -> list[str]:
        """Get all feature names in a group."""
        return [f.name for f in cls._features.values() if f.group == group]
