"""Transaction evaluation model."""

from decimal import Decimal

from pydantic import BaseModel, Field


class TransactionEvaluation(BaseModel):
    """Evaluation metrics for a single transaction. [cite: 119]"""

    amount: Decimal = Field(
        ...,
        decimal_places=2,
        description="Transaction amount [cite: 119]",
        examples=[Decimal("99.99"), Decimal("1250.00"), Decimal("15.47")],
    )

    amount_to_avg_ratio: float = Field(
        ...,
        description=(
            "Ratio of transaction amount to user's average transaction. "
            "Anomaly if > 5.0 [cite: 73]"
        ),
        examples=[1.2, 0.8, 6.5],
    )

    merchant_risk_score: int = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Risk score assigned to the merchant (0-100). Anomaly if > 80 [cite: 73]"
        ),
        examples=[25, 55, 92],
    )

    is_returned: bool = Field(
        ...,
        description="Target label indicating if the transaction was returned",
        examples=[False, True],
    )

    @property
    def is_amount_anomaly(self) -> bool:
        """Check if amount ratio indicates anomaly (> 5.0)."""
        return self.amount_to_avg_ratio > 5.0

    @property
    def is_merchant_anomaly(self) -> bool:
        """Check if merchant risk score indicates anomaly (> 80)."""
        return self.merchant_risk_score > 80
