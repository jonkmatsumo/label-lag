"""Behavior metrics model."""

from decimal import Decimal

from pydantic import BaseModel, Field


class BehaviorMetrics(BaseModel):
    """Behavioral metrics derived from account activity over time."""

    avg_available_balance_30d: Decimal = Field(
        ...,
        decimal_places=2,
        description="Average available balance over the last 30 days",
        examples=[Decimal("2450.00"), Decimal("15000.50"), Decimal("892.33")],
    )

    balance_volatility_z_score: float = Field(
        ...,
        description=(
            "Z-score measuring balance volatility. "
            "High risk indicator if Z < -2.5 [cite: 17, 35]"
        ),
        examples=[0.5, -1.2, -3.1],
    )

    @property
    def is_high_risk_volatility(self) -> bool:
        """Check if balance volatility indicates high risk (Z < -2.5)."""
        return self.balance_volatility_z_score < -2.5
