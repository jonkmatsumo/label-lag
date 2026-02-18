"""Business logic for signal forecasting."""

import os
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import structlog

from forecast.metrics import heuristic_fallback_total, zero_fallback_total
from training.crud_client import get_crud_client
from training.schemas import (
    SignalRequest,
)

logger = structlog.get_logger(__name__)

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


@dataclass(frozen=True)
class ModelPredictionResult:
    """Result payload for model prediction attempts."""

    score: float
    used_fallback: bool = False
    fallback_reason: str | None = None


@dataclass
class SignalForecaster:
    """Forecaster for fraud signals.

    This service is idempotent - it only assesses risk without modifying
    any transaction state. It focuses on pure ML prediction and probability
    calculation, providing inputs for the rule-based decisioning gateway.

    Uses the trained ML model when available, falls back to heuristic-based scoring.
    """

    model_version: str = MODEL_VERSION

    def predict(
        self,
        request: SignalRequest,
        features_override: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> dict:
        """Perform prediction only, skipping rule evaluation.

        Args:
            request: The signal request.
            features_override: Optional pre-hydrated features to use.

        Returns:
            Dict with prediction results.
        """
        import time

        from forecast.model_manager import get_model_manager

        start_time = time.time()
        request_id = request_id or f"req_{uuid.uuid4().hex[:12]}"

        if features_override:
            logger.debug(f"Using pre-hydrated features for user {request.user_id}")
            features = self._map_to_feature_vector(request, features_override)
        else:
            features = self._fetch_features(request)

        manager = get_model_manager()

        fallback_used = False
        fallback_reason = None
        effective_fallback_mode = None
        if manager.model_loaded and features.has_history:
            prediction_result = self._predict_with_model(manager, features)
            if isinstance(prediction_result, ModelPredictionResult):
                raw_probability = prediction_result.score
                fallback_used = prediction_result.used_fallback
                fallback_reason = prediction_result.fallback_reason
            else:
                # Backward-compatible for tests/mocks that patch a raw float.
                raw_probability = float(prediction_result)

            if fallback_used:
                effective_fallback_mode = "probability"
                heuristic_fallback_total.labels(
                    reason=fallback_reason or "model_prediction_error"
                ).inc()

            model_version = manager.model_version
            model_loaded = True
        else:
            # Determine fallback mode: request value -> env var -> default "probability"
            fallback_mode = request.fallback_mode or os.getenv(
                "FORECASTER_FALLBACK_MODE", "probability"
            )
            fallback_reason = (
                "model_not_loaded" if not manager.model_loaded else "no_history"
            )

            if fallback_mode == "error":
                raise RuntimeError(
                    f"Forecaster fallback triggered (mode=error): {fallback_reason}"
                )

            fallback_used = True
            if fallback_mode == "zero":
                raw_probability = 0.0
                zero_fallback_total.labels(reason=fallback_reason).inc()
            elif os.getenv("DISABLE_HEURISTIC_FALLBACK") == "true":
                raw_probability = 0.05
                heuristic_fallback_total.labels(reason="heuristic_disabled").inc()
            else:
                raw_probability = self._calculate_probability(features)
                heuristic_fallback_total.labels(reason=fallback_reason).inc()

            model_version = self.model_version
            model_loaded = False
            effective_fallback_mode = fallback_mode

        # Use calibrator from manager (C2)
        calibrator = manager.calibrator
        score = int(calibrator.transform(np.array([raw_probability]))[0])

        latency_ms = (time.time() - start_time) * 1000

        # Attempt to get coverage diagnostics if we predicted with model
        coverage_info = {}
        if manager.model_loaded and features.has_history:
            # We don't have direct access to internal counts here without refactoring,
            # but we can infer basic presence.
            coverage_info["required_count"] = len(manager.required_features)

        diagnostics = {
            "has_history": features.has_history,
            "raw_probability": float(raw_probability),
            "fallback_used": fallback_used,
            "calibrator_loaded": manager.calibrator_loaded,
            **coverage_info,
        }
        if fallback_reason:
            diagnostics["fallback_reason"] = fallback_reason
        if fallback_used:
            diagnostics["fallback_mode_effective"] = effective_fallback_mode

        response = {
            "request_id": request_id,
            "model_score": score,
            "model_version": model_version,
            "model_loaded": model_loaded,
            "fallback_used": fallback_used,
            "latency_ms": latency_ms,
            "diagnostics": diagnostics,
        }

        if request.include_importance:
            importance = manager.cached_feature_importance
            if importance:
                response["feature_importance"] = importance
            else:
                diagnostics["importance_unavailable"] = True

        # Structured log for every prediction with key diagnostic fields
        logger.info(
            "prediction_complete",
            request_id=request_id,
            user_id=request.user_id,
            model_version=model_version,
            model_loaded=model_loaded,
            fallback=fallback_used,
            fallback_reason=fallback_reason,
            model_source=manager.model_source,
            score=score,
            latency_ms=round(latency_ms, 2),
        )

        return response

    def _predict_with_model(
        self, manager, features: FeatureVector
    ) -> ModelPredictionResult:
        """Use the ML model for prediction.

        Args:
            manager: ModelManager with loaded model.
            features: Feature vector from database.

        Returns:
            Model prediction result with fallback diagnostics.
        """
        try:
            # Get required features from the model
            required_features = manager.required_features

            # Build feature dict using only required features
            # Map from FeatureVector attributes (which may not have all features)
            feature_dict = {}
            available_count = 0
            missing_features = []

            for feature_name in required_features:
                # Try to get attribute from FeatureVector (case-insensitive)
                if hasattr(features, feature_name):
                    val = getattr(features, feature_name)
                    if val is not None:
                        feature_dict[feature_name] = val
                        available_count += 1
                    else:
                        missing_features.append(feature_name)
                        feature_dict[feature_name] = None
                else:
                    missing_features.append(feature_name)
                    feature_dict[feature_name] = None

            # Compute coverage metrics (FF3)
            total_required = len(required_features)
            coverage_ratio = (
                available_count / total_required if total_required > 0 else 1.0
            )

            log_data = {
                "event": "inference_feature_coverage",
                "model_version": manager.model_version,
                "required_count": total_required,
                "available_count": available_count,
                "coverage_ratio": round(coverage_ratio, 4),
            }

            if coverage_ratio < 1.0:
                log_data["missing_features"] = missing_features[:10]
                logger.warning(f"Incomplete feature coverage for inference: {log_data}")
            else:
                logger.info(f"Full feature coverage for inference: {log_data}")

            # Check if any required features are missing
            if missing_features:
                # Still log the failure but provide metrics above
                return ModelPredictionResult(
                    score=self._calculate_probability(features),
                    used_fallback=True,
                    fallback_reason="missing_features",
                )

            probability = manager.predict_single(feature_dict)
            if probability is None:
                # predict_single returned None due to missing features
                return ModelPredictionResult(
                    score=self._calculate_probability(features),
                    used_fallback=True,
                    fallback_reason="missing_features",
                )

            logger.debug(f"ML model prediction: {probability}")
            return ModelPredictionResult(score=float(probability))
        except Exception as e:
            logger.warning(f"ML prediction failed, falling back to heuristic: {e}")
            return ModelPredictionResult(
                score=self._calculate_probability(features),
                used_fallback=True,
                fallback_reason="model_prediction_error",
            )

    def _map_to_feature_vector(
        self, request: SignalRequest, feature_dict: dict[str, Any]
    ) -> FeatureVector:
        """Map a raw dictionary of features to a FeatureVector.

        Args:
            request: The original request.
            feature_dict: Raw feature dictionary.

        Returns:
            Populated FeatureVector.
        """
        return FeatureVector(
            velocity_24h=int(feature_dict.get("velocity_24h", 0)),
            amount_to_avg_ratio_30d=float(
                feature_dict.get("amount_to_avg_ratio_30d", 1.0)
            ),
            balance_volatility_z_score=float(
                feature_dict.get("balance_volatility_z_score", 0.0)
            ),
            bank_connections_24h=int(feature_dict.get("bank_connections_24h", 0)),
            merchant_risk_score=int(feature_dict.get("merchant_risk_score", 0)),
            has_history=feature_dict.get("has_history", True),
            transaction_amount=request.amount,
        )

    def _fetch_features(self, request: SignalRequest) -> FeatureVector:
        """Fetch features for the user from feature store via Analytics service.

        Queries the Analytics service for the most recent features
        for the given user. Falls back to simulated features if not found.

        Args:
            request: The signal request containing user_id.

        Returns:
            FeatureVector with user features.
        """
        try:
            client = get_crud_client()
            tx = client.get_features(user_id=request.user_id)

            if tx is not None:
                logger.debug(f"Found features for user {request.user_id} via Analytics")
                return FeatureVector(
                    velocity_24h=int(tx.velocity_24h),
                    amount_to_avg_ratio_30d=float(tx.amount_to_avg_ratio_30d),
                    balance_volatility_z_score=float(tx.balance_volatility_z_score),
                    bank_connections_24h=0,  # Not in feature_snapshots yet
                    merchant_risk_score=int(tx.merchant_risk_score),
                    has_history=True,
                    transaction_amount=request.amount,
                )
            else:
                logger.debug(
                    f"No features found for user {request.user_id} in Analytics"
                )

        except Exception as e:
            logger.warning(f"Failed to fetch features from Analytics: {e}")

        # Fallback: mark as no history. Do NOT simulate by default.
        # Unknown user behavior is now Go-aligned (either Go hydrates/simulates,
        # or Python uses model fallback logic without features).
        disable_sim = os.getenv("DISABLE_FEATURE_SIMULATION", "true").lower() == "true"
        if disable_sim:
            return FeatureVector(has_history=False, transaction_amount=request.amount)

        return self._simulate_features(request)

    def _simulate_features(self, request: SignalRequest) -> FeatureVector:
        """Generate simulated features for users not in database.

        Args:
            request: The signal request containing user_id.

        Returns:
            FeatureVector with simulated features.
        """
        # Use deterministic hash for consistent results per user
        user_hash = hash(request.user_id) % 1000

        # Simulate feature distribution based on user hash
        velocity = (user_hash % 10) + 1
        amount_ratio = 0.5 + (user_hash % 50) / 10.0
        balance_z = -3.0 + (user_hash % 60) / 10.0
        connections = user_hash % 8
        merchant_risk = user_hash % 100

        return FeatureVector(
            velocity_24h=velocity,
            amount_to_avg_ratio_30d=amount_ratio,
            balance_volatility_z_score=balance_z,
            bank_connections_24h=connections,
            merchant_risk_score=merchant_risk,
            has_history=False,  # Mark as no history since not in DB
            transaction_amount=request.amount,
        )

    def _calculate_probability(self, features: FeatureVector) -> float:
        """Calculate raw fraud probability from features.

        Uses a simple logistic-style combination of risk factors.
        In production, this would be an XGBoost model.

        Args:
            features: The feature vector for forecasting.

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


# Singleton forecaster instance
_forecaster: SignalForecaster | None = None


def get_forecaster() -> SignalForecaster:
    """Get or create the signal forecaster singleton.

    Returns:
        SignalForecaster instance.
    """
    global _forecaster
    if _forecaster is None:
        _forecaster = SignalForecaster()
    return _forecaster
