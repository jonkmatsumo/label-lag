"""Account snapshot model."""

from decimal import Decimal

from pydantic import BaseModel, Field


class AccountSnapshot(BaseModel):
    """Point-in-time snapshot of account state."""

    available_balance: Decimal = Field(
        ...,
        decimal_places=2,
        description="Current available balance in the account",
        examples=[Decimal("1523.47"), Decimal("10000.00"), Decimal("42.89")],
    )

    balance_to_transaction_ratio: float = Field(
        ...,
        description="Ratio of current balance to typical transaction amount",
        examples=[15.3, 2.1, 0.8],
    )
