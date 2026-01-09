"""Business logic for signal evaluation."""

import uuid
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from api.schemas import RiskComponent, SignalRequest, SignalResponse
from model.evaluate import ScoreCalibrator

# Feature thresholds for risk component detection (based on percentiles)
VELOCITY_HIGH_THRESHOLD = 5  # 24h transaction count threshold
AMOUNT_RATIO_HIGH_THRESHOLD = 3.0  # Amount vs 30d avg threshold
BALANCE_VOLATILITY_THRESHOLD = -2.0  # Z-score threshold (negative = low balance)
MERCHANT_RISK_THRESHOLD = 70  # Merchant risk score threshold
CONNECTION_BURST_THRESHOLD = 4  # 24h bank connections threshold

MODEL_VERSION = "v1.0.0"


@dataclass
class FeatureVector:
    """Container for user features used in scoring."""

    velocity_24h: int = 0
    amount_to_avg_ratio_30d: float = 1.0
    balance_volatility_z_score: float = 0.0
    bank_connections_24h: int = 0
    merchant_risk_score: int = 0
    has_history: bool = True
    transaction_amount: Decimal = Decimal("0")


@dataclass
class SignalEvaluator:
    """Evaluates fraud signals for transactions.

    This service is idempotent - it only assesses risk without modifying
    any transaction state.
    """

    calibrator: ScoreCalibrator = field(default_factory=ScoreCalibrator)
    model_version: str = MODEL_VERSION

    def evaluate(self, request: SignalRequest) -> SignalResponse:
        """Evaluate fraud signal for a transaction.

        Args:
            request: The signal evaluation request.

        Returns:
            SignalResponse with score and risk components.
        """
        # Generate unique request ID
        request_id = f"req_{uuid.uuid4().hex[:12]}"

        # Fetch features for the user
        features = self._fetch_features(request)

        # Calculate raw probability from features
        raw_probability = self._calculate_probability(features)

        # Calibrate to 1-99 score
        score = self._calibrate_score(raw_probability)

        # Identify risk components based on feature values
        risk_components = self._identify_risk_components(features)

        return SignalResponse(
            request_id=request_id,
            score=score,
            risk_components=risk_components,
            model_version=self.model_version,
        )

    def _fetch_features(self, request: SignalRequest) -> FeatureVector:
        """Fetch features for the user from feature store.

        In production, this would query the feature_snapshots table.
        For now, we simulate feature retrieval with heuristics.

        Args:
            request: The signal request containing user_id.

        Returns:
            FeatureVector with user features.
        """
        # Simulate feature lookup - in production this queries the DB
        # Use deterministic hash for consistent results per user
        user_hash = hash(request.user_id) % 1000

        # Simulate feature distribution based on user hash
        velocity = (user_hash % 10) + 1
        amount_ratio = 0.5 + (user_hash % 50) / 10.0
        balance_z = -3.0 + (user_hash % 60) / 10.0
        connections = user_hash % 8
        merchant_risk = user_hash % 100
        has_history = user_hash > 100

        return FeatureVector(
            velocity_24h=velocity,
            amount_to_avg_ratio_30d=amount_ratio,
            balance_volatility_z_score=balance_z,
            bank_connections_24h=connections,
            merchant_risk_score=merchant_risk,
            has_history=has_history,
            transaction_amount=request.amount,
        )

    def _calculate_probability(self, features: FeatureVector) -> float:
        """Calculate raw fraud probability from features.

        Uses a simple logistic-style combination of risk factors.
        In production, this would be an XGBoost model.

        Args:
            features: The feature vector for scoring.

        Returns:
            Raw probability between 0.0 and 1.0.
        """
        # Base probability
        prob = 0.05

        # Velocity contribution
        if features.velocity_24h > VELOCITY_HIGH_THRESHOLD:
            prob += 0.15 * min(features.velocity_24h / 10, 1.0)

        # Amount ratio contribution
        if features.amount_to_avg_ratio_30d > AMOUNT_RATIO_HIGH_THRESHOLD:
            ratio_factor = min(features.amount_to_avg_ratio_30d / 10.0, 1.0)
            prob += 0.20 * ratio_factor

        # Balance volatility contribution (negative z-score = low balance)
        if features.balance_volatility_z_score < BALANCE_VOLATILITY_THRESHOLD:
            vol_factor = min(abs(features.balance_volatility_z_score) / 5.0, 1.0)
            prob += 0.15 * vol_factor

        # Connection burst contribution
        if features.bank_connections_24h > CONNECTION_BURST_THRESHOLD:
            conn_factor = min(features.bank_connections_24h / 10.0, 1.0)
            prob += 0.20 * conn_factor

        # Merchant risk contribution
        if features.merchant_risk_score > MERCHANT_RISK_THRESHOLD:
            merchant_factor = (features.merchant_risk_score - 70) / 30.0
            prob += 0.15 * min(merchant_factor, 1.0)

        # Insufficient history penalty
        if not features.has_history:
            prob += 0.10

        # Cap probability
        return min(prob, 0.99)

    def _calibrate_score(self, probability: float) -> int:
        """Convert probability to calibrated 1-99 score.

        Args:
            probability: Raw probability between 0.0 and 1.0.

        Returns:
            Integer score between 1 and 99.
        """
        prob_array = np.array([probability])
        scores = self.calibrator.transform(prob_array)
        return int(scores[0])

    def _identify_risk_components(self, features: FeatureVector) -> list[RiskComponent]:
        """Identify risk components based on feature thresholds.

        This provides interpretability by flagging which features
        contributed to a high score.

        Args:
            features: The feature vector.

        Returns:
            List of risk components that triggered.
        """
        components = []

        if features.velocity_24h > VELOCITY_HIGH_THRESHOLD:
            components.append(
                RiskComponent(
                    key="velocity",
                    label="high_transaction_velocity",
                )
            )

        if features.amount_to_avg_ratio_30d > AMOUNT_RATIO_HIGH_THRESHOLD:
            components.append(
                RiskComponent(
                    key="amount_ratio",
                    label="unusual_transaction_amount",
                )
            )

        if features.balance_volatility_z_score < BALANCE_VOLATILITY_THRESHOLD:
            components.append(
                RiskComponent(
                    key="balance",
                    label="low_balance_volatility",
                )
            )

        if features.bank_connections_24h > CONNECTION_BURST_THRESHOLD:
            components.append(
                RiskComponent(
                    key="connections",
                    label="connection_burst_detected",
                )
            )

        if features.merchant_risk_score > MERCHANT_RISK_THRESHOLD:
            components.append(
                RiskComponent(
                    key="merchant",
                    label="high_risk_merchant",
                )
            )

        if not features.has_history:
            components.append(
                RiskComponent(
                    key="history",
                    label="insufficient_history",
                )
            )

        return components


# Singleton evaluator instance
_evaluator: SignalEvaluator | None = None


def get_evaluator() -> SignalEvaluator:
    """Get or create the signal evaluator singleton.

    Returns:
        SignalEvaluator instance.
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = SignalEvaluator()
    return _evaluator
