"""FastAPI application for fraud signal evaluation.

This API provides idempotent risk assessment for transactions.
It does not modify transaction state - it only provides an evaluation.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from api.drift_cache import get_drift_cache
from api.model_manager import get_model_manager
from api.schemas import (
    AcceptSuggestionRequest,
    AcceptSuggestionResponse,
    ActivateRuleRequest,
    ActivateRuleResponse,
    ApproveRuleRequest,
    ApproveRuleResponse,
    AuditLogQueryResponse,
    AuditRecordResponse,
    BacktestMetricsResponse,
    BacktestResultResponse,
    BacktestResultsListResponse,
    ClearDataResponse,
    ConflictResponse,
    DisableRuleRequest,
    DisableRuleResponse,
    DraftRuleCreateRequest,
    DraftRuleCreateResponse,
    DraftRuleListResponse,
    DraftRuleResponse,
    DraftRuleSubmitRequest,
    DraftRuleSubmitResponse,
    DraftRuleUpdateRequest,
    DraftRuleUpdateResponse,
    DraftRuleValidateRequest,
    DraftRuleValidateResponse,
    DriftStatusResponse,
    FeatureDriftDetail,
    GenerateDataRequest,
    GenerateDataResponse,
    HealthResponse,
    RedundancyResponse,
    RejectRuleRequest,
    RejectRuleResponse,
    RollbackRuleRequest,
    RollbackRuleResponse,
    RuleMetricsItem,
    RuleSuggestionResponse,
    RuleVersionListResponse,
    RuleVersionResponse,
    SandboxEvaluateRequest,
    SandboxEvaluateResponse,
    SandboxMatchedRule,
    ShadowComparisonResponse,
    ShadowRuleRequest,
    ShadowRuleResponse,
    SignalRequest,
    SignalResponse,
    SuggestionEvidence,
    SuggestionsListResponse,
    TrainRequest,
    TrainResponse,
    ValidationResult,
)
from api.services import get_evaluator

if TYPE_CHECKING:
    from synthetic_pipeline.db.models import EvaluationMetadataDB, GeneratedRecordDB
    from synthetic_pipeline.models import EvaluationMetadata, GeneratedRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _pydantic_to_db(record: "GeneratedRecord") -> "GeneratedRecordDB":
    """Convert a Pydantic GeneratedRecord to SQLAlchemy model."""
    from synthetic_pipeline.db.models import GeneratedRecordDB

    return GeneratedRecordDB(
        record_id=record.record_id,
        user_id=record.user_id,
        full_name=record.full_name,
        email=record.email,
        phone=record.phone,
        transaction_timestamp=record.transaction_timestamp,
        is_off_hours_txn=record.is_off_hours_txn,
        available_balance=record.account.available_balance,
        balance_to_transaction_ratio=record.account.balance_to_transaction_ratio,
        avg_available_balance_30d=record.behavior.avg_available_balance_30d,
        balance_volatility_z_score=record.behavior.balance_volatility_z_score,
        bank_connections_count_24h=record.connection.bank_connections_count_24h,
        bank_connections_count_7d=record.connection.bank_connections_count_7d,
        bank_connections_avg_30d=record.connection.bank_connections_avg_30d,
        amount=record.transaction.amount,
        amount_to_avg_ratio=record.transaction.amount_to_avg_ratio,
        merchant_risk_score=record.transaction.merchant_risk_score,
        is_returned=record.transaction.is_returned,
        email_changed_at=record.identity_changes.email_changed_at,
        phone_changed_at=record.identity_changes.phone_changed_at,
        is_fraudulent=record.is_fraudulent,
        fraud_type=record.fraud_type,
    )


def _metadata_to_db(meta: "EvaluationMetadata") -> "EvaluationMetadataDB":
    """Convert a Pydantic EvaluationMetadata to SQLAlchemy model."""
    from synthetic_pipeline.db.models import EvaluationMetadataDB

    return EvaluationMetadataDB(
        user_id=meta.user_id,
        record_id=meta.record_id,
        sequence_number=meta.sequence_number,
        fraud_confirmed_at=meta.fraud_confirmed_at,
        is_pre_fraud=meta.is_pre_fraud,
        days_to_fraud=meta.days_to_fraud,
        is_train_eligible=meta.is_train_eligible,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - load model on startup."""
    # Startup: Load the production model
    logger.info("Starting up - loading production model...")
    manager = get_model_manager()
    success = manager.load_production_model()

    if success:
        logger.info(
            f"Model loaded successfully: version={manager.model_version}, "
            f"source={manager.model_source}"
        )
    else:
        logger.warning("No model loaded - API will use rule-based evaluation only")

    yield

    # Shutdown: cleanup if needed
    logger.info("Shutting down...")


app = FastAPI(
    title="Fraud Signal API",
    description="Risk signal evaluation for fraud detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check API health status.

    Returns:
        HealthResponse with status and model information.
    """
    manager = get_model_manager()
    evaluator = get_evaluator()

    # Use model manager version if available, otherwise fall back to evaluator
    version = manager.model_version if manager.model_loaded else evaluator.model_version

    return HealthResponse(
        status="healthy",
        model_loaded=manager.model_loaded,
        version=version,
    )


@app.get(
    "/monitoring/drift",
    response_model=DriftStatusResponse,
    tags=["Monitoring"],
    summary="Check dataset drift status",
)
async def get_drift_status(
    hours: int = Query(
        default=24, ge=1, le=168, description="Hours of live data to analyze"
    ),
    force_refresh: bool = Query(
        default=False, description="Bypass cache and recompute"
    ),
) -> DriftStatusResponse:
    """Check feature distribution drift between reference and live data.

    Compares reference data from the production model with recent live data
    using Population Stability Index (PSI) to detect distribution shifts.

    Args:
        hours: Number of hours of live data to analyze (1-168).
        force_refresh: If True, bypass cache and recompute drift.

    Returns:
        DriftStatusResponse with drift status, top features, and metadata.
    """
    import time

    from monitor.detect_drift import (
        PSI_THRESHOLD_CRITICAL,
        PSI_THRESHOLD_WARNING,
        detect_drift,
    )

    start_time = time.time()
    cache = get_drift_cache()
    threshold = PSI_THRESHOLD_CRITICAL

    # Check cache unless force_refresh is True
    cached_result = None
    if not force_refresh:
        cached_result = cache.get(hours, threshold)

    if cached_result is not None:
        # Return cached result
        duration_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "Drift check completed (cached)",
            extra={
                "hours": hours,
                "cached": True,
                "duration_ms": duration_ms,
            },
        )
        return _build_drift_response(cached_result, cached=True)

    # Compute fresh result
    try:
        result = detect_drift(hours=hours, threshold=threshold)
        cache.set(hours, threshold, result)

        duration_ms = int((time.time() - start_time) * 1000)
        features_evaluated = len(result.get("features", {}))

        # Determine overall status
        overall_status = "unknown"
        if "error" in result:
            overall_status = "unknown"
        elif result.get("drift_detected", False):
            overall_status = "fail"
        else:
            # Check if any feature has WARNING status
            has_warning = any(
                details.get("status") == "WARNING"
                for details in result.get("features", {}).values()
            )
            if has_warning:
                overall_status = "warn"
            else:
                overall_status = "ok"

        logger.info(
            "Drift check completed",
            extra={
                "hours": hours,
                "cached": False,
                "status": overall_status,
                "duration_ms": duration_ms,
                "features_evaluated": features_evaluated,
            },
        )

        return _build_drift_response(
            result, cached=False, overall_status=overall_status
        )

    except Exception as e:
        logger.exception("Drift detection failed", extra={"hours": hours})
        return DriftStatusResponse(
            status="unknown",
            computed_at=datetime.now(timezone.utc).isoformat(),
            cached=False,
            reference_window="Unknown",
            current_window=f"Last {hours} hours",
            reference_size=0,
            live_size=0,
            top_features=[],
            thresholds={
                "warn": PSI_THRESHOLD_WARNING,
                "fail": PSI_THRESHOLD_CRITICAL,
            },
            error=str(e),
        )


def _build_drift_response(
    result: dict,
    cached: bool,
    overall_status: str | None = None,
) -> DriftStatusResponse:
    """Build DriftStatusResponse from detect_drift result.

    Args:
        result: Result dict from detect_drift().
        cached: Whether result was from cache.
        overall_status: Overall status (ok/warn/fail/unknown). If None, computed.

    Returns:
        DriftStatusResponse instance.
    """
    from monitor.detect_drift import (
        PSI_THRESHOLD_CRITICAL,
        PSI_THRESHOLD_WARNING,
    )

    # Determine overall status if not provided
    if overall_status is None:
        if "error" in result:
            overall_status = "unknown"
        elif result.get("drift_detected", False):
            overall_status = "fail"
        else:
            # Check if any feature has WARNING status
            has_warning = any(
                details.get("status") == "WARNING"
                for details in result.get("features", {}).values()
            )
            if has_warning:
                overall_status = "warn"
            else:
                overall_status = "ok"

    # Build top features list (sorted by PSI descending)
    top_features = []
    for feature_name, details in result.get("features", {}).items():
        top_features.append(
            FeatureDriftDetail(
                feature=feature_name,
                psi=details.get("psi", 0.0),
                status=details.get("status", "OK"),
            )
        )
    # Sort by PSI descending
    top_features.sort(key=lambda x: x.psi, reverse=True)

    # Build reference window description
    reference_window = "Production model reference data"
    if result.get("reference_size", 0) > 0:
        reference_window = (
            f"Production model reference data ({result['reference_size']} samples)"
        )

    # Build current window description
    hours_analyzed = result.get("hours_analyzed", 24)
    current_window = f"Last {hours_analyzed} hours"
    if result.get("live_size", 0) > 0:
        current_window = f"Last {hours_analyzed} hours ({result['live_size']} samples)"

    return DriftStatusResponse(
        status=overall_status,
        computed_at=result.get("timestamp", datetime.now(timezone.utc).isoformat()),
        cached=cached,
        reference_window=reference_window,
        current_window=current_window,
        reference_size=result.get("reference_size", 0),
        live_size=result.get("live_size", 0),
        top_features=top_features,
        thresholds={
            "warn": PSI_THRESHOLD_WARNING,
            "fail": PSI_THRESHOLD_CRITICAL,
        },
        error=result.get("error"),
    )


@app.post("/reload-model", tags=["System"])
async def reload_model() -> dict:
    """Reload the production model from MLflow.

    Call this endpoint after promoting a new model to production
    to pick up the latest version without restarting the API.

    Returns:
        Dict with success status and model version.
    """
    manager = get_model_manager()
    success = manager.load_production_model()

    if success:
        logger.info(f"Model reloaded: version={manager.model_version}")
        return {
            "success": True,
            "model_loaded": True,
            "version": manager.model_version,
            "source": manager.model_source,
        }
    else:
        logger.warning("Model reload failed")
        return {
            "success": False,
            "model_loaded": False,
            "version": None,
            "source": "none",
        }


@app.post(
    "/evaluate/signal",
    response_model=SignalResponse,
    tags=["Evaluation"],
    summary="Evaluate fraud signal",
    description="""
Evaluate the fraud risk signal for a transaction.

This endpoint is **idempotent** - it only provides an assessment without
modifying any transaction state. The same input will produce consistent
scoring (deterministic per user_id).

The response includes:
- **score**: Risk score from 1 (lowest) to 99 (highest)
- **risk_components**: Factors contributing to the score
- **model_version**: Version of the scoring model

Scores are calibrated to match real-world fraud score distributions where:
- Scores 1-20 are very common (low risk)
- Scores 80+ are rare (high risk)
""",
    responses={
        200: {
            "description": "Successful evaluation",
            "content": {
                "application/json": {
                    "example": {
                        "request_id": "req_123xyz",
                        "score": 85,
                        "risk_components": [
                            {
                                "key": "velocity",
                                "label": "high_transaction_velocity",
                            },
                            {"key": "history", "label": "insufficient_history"},
                        ],
                        "model_version": "v1.0.0",
                    }
                }
            },
        },
        422: {"description": "Validation error"},
    },
)
async def evaluate_signal(request: SignalRequest) -> SignalResponse:
    """Evaluate fraud signal for a transaction.

    Args:
        request: Signal evaluation request with user_id, amount, currency,
            and client_transaction_id.

    Returns:
        SignalResponse with risk score and contributing factors.
    """
    try:
        evaluator = get_evaluator()
        return evaluator.evaluate(request)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {e!s}",
        ) from e


@app.post(
    "/data/generate",
    response_model=GenerateDataResponse,
    tags=["Data Management"],
    summary="Generate synthetic data",
    description="Generate synthetic transaction data with configurable fraud rate.",
)
async def generate_data(request: GenerateDataRequest) -> GenerateDataResponse:
    """Generate synthetic transaction data.

    Args:
        request: Generation request with num_users, fraud_rate, and drop_existing.

    Returns:
        GenerateDataResponse with counts of generated records.
    """
    try:
        from pipeline.materialize_features import FeatureMaterializer
        from synthetic_pipeline.db.models import Base
        from synthetic_pipeline.db.session import DatabaseSession
        from synthetic_pipeline.generator import DataGenerator

        # Generate data
        generator = DataGenerator()
        result = generator.generate_dataset_with_sequences(
            num_users=request.num_users,
            fraud_rate=request.fraud_rate,
        )

        # Count fraud records
        fraud_count = sum(1 for r in result.records if r.is_fraudulent)

        # Connect to database
        db_session = DatabaseSession()

        with db_session.get_session() as session:
            if request.drop_existing:
                # Drop and recreate tables
                Base.metadata.drop_all(db_session.engine)
                Base.metadata.create_all(db_session.engine)
            else:
                # Just ensure tables exist
                Base.metadata.create_all(db_session.engine)

            # Convert and insert records
            db_records = [_pydantic_to_db(record) for record in result.records]
            session.bulk_save_objects(db_records)

            # Insert metadata
            meta_records = [_metadata_to_db(meta) for meta in result.metadata]
            session.bulk_save_objects(meta_records)

            session.commit()

        # Materialize features
        materializer = FeatureMaterializer()
        materialize_stats = materializer.materialize_all()
        features_count = materialize_stats.get("total_processed", 0)

        return GenerateDataResponse(
            success=True,
            total_records=len(result.records),
            fraud_records=fraud_count,
            features_materialized=features_count,
        )

    except Exception as e:
        logger.exception("Data generation failed")
        return GenerateDataResponse(success=False, error=str(e))


@app.delete(
    "/data/clear",
    response_model=ClearDataResponse,
    tags=["Data Management"],
    summary="Clear all data",
    description="Delete all records from the database tables.",
)
async def clear_data() -> ClearDataResponse:
    """Clear all data from the database.

    Returns:
        ClearDataResponse with list of cleared tables.
    """
    try:
        from synthetic_pipeline.db.models import Base
        from synthetic_pipeline.db.session import DatabaseSession

        db_session = DatabaseSession()

        # Get table names before dropping
        table_names = [table.name for table in Base.metadata.sorted_tables]

        # Drop all tables
        Base.metadata.drop_all(db_session.engine)

        # Recreate empty tables
        Base.metadata.create_all(db_session.engine)

        return ClearDataResponse(
            success=True,
            tables_cleared=table_names,
        )

    except Exception as e:
        logger.exception("Data clearing failed")
        return ClearDataResponse(success=False, error=str(e))


@app.post(
    "/train",
    response_model=TrainResponse,
    tags=["Training"],
    summary="Train a new model",
    description="Train a new XGBoost model with the specified hyperparameters.",
)
async def train_model_endpoint(request: TrainRequest) -> TrainResponse:
    """Train a new model with specified parameters.

    Args:
        request: Training request with max_depth and training_window_days.

    Returns:
        TrainResponse with success status and run_id or error.
    """
    try:
        from model.train import train_model

        run_id = train_model(
            max_depth=request.max_depth,
            training_window_days=request.training_window_days,
            feature_columns=request.selected_feature_columns,
            split_config=request.split_config,
            n_estimators=request.n_estimators,
            learning_rate=request.learning_rate,
            min_child_weight=request.min_child_weight,
            subsample=request.subsample,
            colsample_bytree=request.colsample_bytree,
            gamma=request.gamma,
            reg_alpha=request.reg_alpha,
            reg_lambda=request.reg_lambda,
            random_state=request.random_state,
            early_stopping_rounds=request.early_stopping_rounds,
            tuning_config=request.tuning_config,
        )
        return TrainResponse(success=True, run_id=run_id)
    except ValueError as e:
        return TrainResponse(success=False, error=str(e))
    except Exception as e:
        logger.exception("Training failed")
        return TrainResponse(success=False, error=str(e))


# =============================================================================
# Rule Inspector Endpoints (Phase 1 - Read-Only/Deterministic)
# =============================================================================


@app.get(
    "/rules",
    tags=["Rule Inspector"],
    summary="List current production ruleset",
    description="Returns the current production ruleset. Read-only endpoint.",
)
async def get_rules() -> dict:
    """Get the current production ruleset.

    Returns:
        Dict containing the ruleset version and rules.
    """
    manager = get_model_manager()
    ruleset = manager.ruleset

    if ruleset is None:
        return {"version": "none", "rules": []}

    return {
        "version": ruleset.version,
        "rules": [
            {
                "id": rule.id,
                "field": rule.field,
                "op": rule.op,
                "value": rule.value,
                "action": rule.action,
                "score": rule.score,
                "severity": rule.severity,
                "reason": rule.reason,
                "status": rule.status,
            }
            for rule in ruleset.rules
        ],
    }


@app.post(
    "/rules/sandbox/evaluate",
    response_model=SandboxEvaluateResponse,
    tags=["Rule Inspector"],
    summary="Evaluate rules in sandbox mode",
    description="""
Deterministic rule evaluation for testing purposes.

This endpoint is **pure function** - no database writes, no model inference,
no production ruleset modification. Safe for experimentation.

Provide feature values and optionally a custom ruleset to test rule behavior
without affecting production.
""",
)
async def sandbox_evaluate(request: SandboxEvaluateRequest) -> SandboxEvaluateResponse:
    """Evaluate rules against features in sandbox mode.

    Args:
        request: Sandbox evaluation request with features, base_score,
            and optional ruleset.

    Returns:
        SandboxEvaluateResponse with evaluation results.
    """
    from api.rules import Rule, RuleSet, evaluate_rules

    # Build feature dict from request
    features = {
        "velocity_24h": request.features.velocity_24h,
        "amount_to_avg_ratio_30d": request.features.amount_to_avg_ratio_30d,
        "balance_volatility_z_score": request.features.balance_volatility_z_score,
        "bank_connections_24h": request.features.bank_connections_24h,
        "merchant_risk_score": request.features.merchant_risk_score,
        "has_history": request.features.has_history,
        "transaction_amount": request.features.transaction_amount,
    }

    # Use custom ruleset if provided, otherwise use production
    if request.ruleset is not None:
        try:
            rules = [
                Rule(
                    id=r.id,
                    field=r.field,
                    op=r.op,
                    value=r.value,
                    action=r.action,
                    score=r.score,
                    severity=r.severity,
                    reason=r.reason,
                    status=r.status,
                )
                for r in request.ruleset.rules
            ]
            ruleset = RuleSet(version=request.ruleset.version, rules=rules)
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid ruleset: {e}") from e
    else:
        manager = get_model_manager()
        ruleset = manager.ruleset
        if ruleset is None:
            ruleset = RuleSet.empty()

    # Evaluate rules
    result = evaluate_rules(features, request.base_score, ruleset)

    # Build matched rules with full info
    matched_rules = []
    for exp in result.explanations:
        rule_id = exp["rule_id"]
        # Find the rule to get action and score
        matching_rule = next((r for r in ruleset.rules if r.id == rule_id), None)
        matched_rules.append(
            SandboxMatchedRule(
                rule_id=rule_id,
                severity=exp["severity"],
                reason=exp["reason"],
                action=matching_rule.action if matching_rule else "",
                score=matching_rule.score if matching_rule else None,
            )
        )

    # Build shadow matched rules
    shadow_matched_rules = []
    for exp in result.shadow_explanations or []:
        rule_id = exp["rule_id"]
        matching_rule = next((r for r in ruleset.rules if r.id == rule_id), None)
        shadow_matched_rules.append(
            SandboxMatchedRule(
                rule_id=rule_id,
                severity=exp["severity"],
                reason=exp["reason"],
                action=matching_rule.action if matching_rule else "",
                score=matching_rule.score if matching_rule else None,
            )
        )

    return SandboxEvaluateResponse(
        final_score=result.final_score,
        matched_rules=matched_rules,
        explanations=result.explanations,
        shadow_matched_rules=shadow_matched_rules,
        rejected=result.rejected,
        ruleset_version=ruleset.version,
    )


@app.get(
    "/metrics/shadow/comparison",
    response_model=ShadowComparisonResponse,
    tags=["Rule Inspector"],
    summary="Get shadow mode comparison metrics",
    description="""
Read-only metrics comparing production vs shadow rule performance.

Returns per-rule match counts and overlap statistics for the specified
date range. No side effects.
""",
)
async def get_shadow_comparison(
    start_date: str = Query(
        ...,
        description="Start date (ISO format, e.g., 2024-01-01)",
    ),
    end_date: str = Query(
        ...,
        description="End date (ISO format, e.g., 2024-01-31)",
    ),
    rule_ids: str | None = Query(
        None,
        description="Comma-separated rule IDs to filter (optional)",
    ),
) -> ShadowComparisonResponse:
    """Get shadow mode comparison metrics.

    Args:
        start_date: Start of date range (ISO format).
        end_date: End of date range (ISO format).
        rule_ids: Optional comma-separated rule IDs to filter.

    Returns:
        ShadowComparisonResponse with comparison metrics.
    """
    from api.metrics import get_metrics_collector

    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Use ISO format (YYYY-MM-DD): {e}",
        ) from e

    # Parse rule_ids if provided
    if rule_ids:
        rule_id_list = [r.strip() for r in rule_ids.split(",") if r.strip()]
    else:
        # Get all rule IDs from the current ruleset
        manager = get_model_manager()
        ruleset = manager.ruleset
        if ruleset:
            rule_id_list = [r.id for r in ruleset.rules]
        else:
            rule_id_list = []

    if not rule_id_list:
        return ShadowComparisonResponse(
            period_start=start_date,
            period_end=end_date,
            rule_metrics=[],
            total_requests=0,
        )

    # Get metrics
    collector = get_metrics_collector()
    report = collector.generate_comparison_report(rule_id_list, start_dt, end_dt)

    # Convert to response
    rule_metrics = [
        RuleMetricsItem(
            rule_id=rm.rule_id,
            period_start=rm.period_start.isoformat(),
            period_end=rm.period_end.isoformat(),
            production_matches=rm.production_matches,
            shadow_matches=rm.shadow_matches,
            overlap_count=rm.overlap_count,
            production_only_count=rm.production_only_count,
            shadow_only_count=rm.shadow_only_count,
        )
        for rm in report.rule_metrics
    ]

    return ShadowComparisonResponse(
        period_start=report.period_start.isoformat(),
        period_end=report.period_end.isoformat(),
        rule_metrics=rule_metrics,
        total_requests=report.total_requests,
    )


@app.get(
    "/backtest/results",
    response_model=BacktestResultsListResponse,
    tags=["Rule Inspector"],
    summary="List backtest results",
    description="""
Read-only list of completed backtest results.

Returns backtest results with optional filters. No side effects.
""",
)
async def list_backtest_results(
    rule_id: str | None = Query(None, description="Filter by rule ID"),
    start_date: str | None = Query(
        None,
        description="Filter results completed after this date (ISO format)",
    ),
    end_date: str | None = Query(
        None,
        description="Filter results completed before this date (ISO format)",
    ),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
) -> BacktestResultsListResponse:
    """List backtest results with optional filters.

    Args:
        rule_id: Optional rule ID filter.
        start_date: Optional start date filter.
        end_date: Optional end date filter.
        limit: Maximum results to return.

    Returns:
        BacktestResultsListResponse with list of results.
    """
    from api.backtest import BacktestStore

    store = BacktestStore()

    # Parse dates if provided
    start_dt = None
    end_dt = None
    try:
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Use ISO format (YYYY-MM-DD): {e}",
        ) from e

    # Get results
    results = store.list_results(
        rule_id=rule_id,
        start_date=start_dt,
        end_date=end_dt,
    )

    # Apply limit
    results = results[:limit]

    # Convert to response
    response_results = [
        BacktestResultResponse(
            job_id=r.job_id,
            rule_id=r.rule_id,
            ruleset_version=r.ruleset_version,
            start_date=r.start_date.isoformat(),
            end_date=r.end_date.isoformat(),
            metrics=BacktestMetricsResponse(
                total_records=r.metrics.total_records,
                matched_count=r.metrics.matched_count,
                match_rate=r.metrics.match_rate,
                score_distribution=r.metrics.score_distribution,
                score_mean=r.metrics.score_mean,
                score_std=r.metrics.score_std,
                score_min=r.metrics.score_min,
                score_max=r.metrics.score_max,
                rejected_count=r.metrics.rejected_count,
                rejected_rate=r.metrics.rejected_rate,
            ),
            completed_at=r.completed_at.isoformat(),
            error=r.error,
        )
        for r in results
    ]

    return BacktestResultsListResponse(
        results=response_results,
        total=len(response_results),
    )


@app.get(
    "/backtest/results/{job_id}",
    response_model=BacktestResultResponse,
    tags=["Rule Inspector"],
    summary="Get backtest result by job ID",
    description="""
Read-only retrieval of a specific backtest result.

Returns the full backtest result for the given job ID. No side effects.
""",
)
async def get_backtest_result(job_id: str) -> BacktestResultResponse:
    """Get a specific backtest result by job ID.

    Args:
        job_id: Backtest job identifier.

    Returns:
        BacktestResultResponse with full result details.

    Raises:
        HTTPException: If result not found.
    """
    from api.backtest import BacktestStore

    store = BacktestStore()
    result = store.get(job_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Backtest result not found: {job_id}",
        )

    return BacktestResultResponse(
        job_id=result.job_id,
        rule_id=result.rule_id,
        ruleset_version=result.ruleset_version,
        start_date=result.start_date.isoformat(),
        end_date=result.end_date.isoformat(),
        metrics=BacktestMetricsResponse(
            total_records=result.metrics.total_records,
            matched_count=result.metrics.matched_count,
            match_rate=result.metrics.match_rate,
            score_distribution=result.metrics.score_distribution,
            score_mean=result.metrics.score_mean,
            score_std=result.metrics.score_std,
            score_min=result.metrics.score_min,
            score_max=result.metrics.score_max,
            rejected_count=result.metrics.rejected_count,
            rejected_rate=result.metrics.rejected_rate,
        ),
        completed_at=result.completed_at.isoformat(),
        error=result.error,
    )


@app.get(
    "/suggestions/heuristic",
    response_model=SuggestionsListResponse,
    tags=["Rule Inspector"],
    summary="Get heuristic rule suggestions",
    description="""
Read-only rule suggestions based on feature distribution analysis.

Analyzes feature distributions and suggests potential rules.
These are suggestions only - no rules are created or modified.
""",
)
async def get_heuristic_suggestions(
    field: str | None = Query(
        None,
        description="Filter by feature field (e.g., velocity_24h)",
    ),
    min_confidence: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    ),
    min_samples: int = Query(
        100,
        ge=10,
        le=10000,
        description="Minimum samples required for analysis",
    ),
) -> SuggestionsListResponse:
    """Get heuristic rule suggestions.

    Args:
        field: Optional field to filter suggestions.
        min_confidence: Minimum confidence threshold.
        min_samples: Minimum samples required.

    Returns:
        SuggestionsListResponse with list of suggestions.
    """
    from api.suggestions import SuggestionEngine

    try:
        engine = SuggestionEngine(min_confidence=min_confidence)
        suggestions = engine.generate_suggestions(field=field, min_samples=min_samples)

        # Convert to response (limit to 50)
        response_suggestions = []
        for s in suggestions[:50]:
            evidence = s.evidence
            response_suggestions.append(
                RuleSuggestionResponse(
                    field=s.field,
                    operator=s.operator,
                    threshold=s.threshold,
                    action=s.action,
                    suggested_score=s.suggested_score,
                    confidence=s.confidence,
                    evidence=SuggestionEvidence(
                        statistic=evidence.get("statistic", ""),
                        value=evidence.get("value", 0.0),
                        mean=evidence.get("mean", 0.0),
                        std=evidence.get("std", 0.0),
                        sample_count=evidence.get("sample_count", 0),
                    ),
                    reason=s.reason,
                )
            )

        return SuggestionsListResponse(
            suggestions=response_suggestions,
            total=len(response_suggestions),
        )

    except Exception as e:
        logger.warning(f"Suggestion generation failed: {e}")
        return SuggestionsListResponse(suggestions=[], total=0)


@app.post(
    "/suggestions/accept",
    response_model=AcceptSuggestionResponse,
    tags=["Rule Inspector"],
    summary="Accept a suggestion as a draft rule",
    description="""
Convert a heuristic/model-assisted suggestion into a draft rule.

Preserves suggestion metadata (confidence, evidence) in audit trail
for traceability. All accepted rules start in draft status.
""",
)
async def accept_suggestion(
    request: AcceptSuggestionRequest,
) -> AcceptSuggestionResponse:
    """Accept a suggestion and create a draft rule.

    The UI should pass the full suggestion data from the suggestions list.
    Optional edits can override any field before creating the draft rule.

    Args:
        request: Accept request with suggestion data, actor, optional custom_id
            and edits.

    Returns:
        AcceptSuggestionResponse with created draft rule and source metadata.

    Raises:
        HTTPException: If rule creation fails or rule ID already exists.
    """
    from api.audit import get_audit_logger
    from api.draft_store import get_draft_store
    from api.suggestions import RuleSuggestion
    from api.versioning import get_version_store

    store = get_draft_store()
    version_store = get_version_store()
    audit_logger = get_audit_logger()

    # Reconstruct RuleSuggestion from response data
    suggestion_data = request.suggestion
    # Convert Pydantic model to dict
    if hasattr(suggestion_data.evidence, "model_dump"):
        evidence_dict = suggestion_data.evidence.model_dump()
    elif hasattr(suggestion_data.evidence, "dict"):
        evidence_dict = suggestion_data.evidence.dict()
    else:
        evidence_dict = dict(suggestion_data.evidence)

    suggestion = RuleSuggestion(
        field=suggestion_data.field,
        operator=suggestion_data.operator,
        threshold=suggestion_data.threshold,
        action=suggestion_data.action,
        suggested_score=suggestion_data.suggested_score,
        confidence=suggestion_data.confidence,
        evidence=evidence_dict,
        reason=suggestion_data.reason,
    )

    # Apply edits if provided
    if request.edits:
        if "field" in request.edits:
            suggestion.field = request.edits["field"]
        if "operator" in request.edits:
            suggestion.operator = request.edits["operator"]
        if "threshold" in request.edits:
            suggestion.threshold = request.edits["threshold"]
        if "action" in request.edits:
            suggestion.action = request.edits["action"]
        if "suggested_score" in request.edits:
            suggestion.suggested_score = request.edits["suggested_score"]
        if "reason" in request.edits:
            suggestion.reason = request.edits["reason"]

    # Convert to rule
    rule_id = request.custom_id
    rule = suggestion.to_rule(rule_id=rule_id)

    # Check if rule ID already exists
    if store.exists(rule.id):
        raise HTTPException(
            status_code=409,
            detail=f"Rule with ID '{rule.id}' already exists",
        )

    # Save to draft store
    try:
        store.save(rule)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Create version snapshot
    version_store.save(
        rule=rule,
        created_by=request.actor,
        reason=f"Accepted suggestion with confidence {suggestion.confidence:.2f}",
    )

    # Create audit record with suggestion metadata
    after_state = {
        "id": rule.id,
        "field": rule.field,
        "op": rule.op,
        "value": rule.value,
        "action": rule.action,
        "score": rule.score,
        "severity": rule.severity,
        "reason": rule.reason,
        "status": rule.status,
    }

    audit_reason = (
        f"Accepted suggestion: {suggestion.field} {suggestion.operator} "
        f"{suggestion.threshold} (confidence: {suggestion.confidence:.2f}, "
        f"evidence: {evidence_dict.get('statistic', 'N/A')})"
    )

    audit_logger.log(
        rule_id=rule.id,
        action="create",
        actor=request.actor,
        before_state=None,
        after_state=after_state,
        reason=audit_reason,
    )

    # Convert to response
    rule_response = DraftRuleResponse(
        rule_id=rule.id,
        field=rule.field,
        op=rule.op,
        value=rule.value,
        action=rule.action,
        score=rule.score,
        severity=rule.severity,
        reason=rule.reason,
        status=rule.status,
        created_at=None,
    )

    source_suggestion = {
        "confidence": suggestion.confidence,
        "evidence": evidence_dict,
        "field": suggestion.field,
        "threshold": suggestion.threshold,
    }

    return AcceptSuggestionResponse(
        rule=rule_response,
        rule_id=rule.id,
        source_suggestion=source_suggestion,
    )


# =============================================================================
# Draft Rule Endpoints
# =============================================================================


@app.post(
    "/rules/draft",
    response_model=DraftRuleCreateResponse,
    tags=["Draft Rules"],
    summary="Create a new draft rule",
    description="""
Create a new rule in draft status.

The rule will be validated for schema correctness, conflicts, and
redundancies. Validation results are returned but do not block creation.
""",
)
async def create_draft_rule(request: DraftRuleCreateRequest) -> DraftRuleCreateResponse:
    """Create a new draft rule.

    Args:
        request: Draft rule creation request with all rule fields and actor.

    Returns:
        DraftRuleCreateResponse with created rule and validation results.

    Raises:
        HTTPException: If rule creation fails or rule ID already exists.
    """
    from api.draft_store import get_draft_store
    from api.rules import Rule, RuleStatus
    from api.validation import validate_ruleset

    store = get_draft_store()

    # Check if rule ID already exists
    if store.exists(request.id):
        raise HTTPException(
            status_code=409,
            detail=f"Rule with ID '{request.id}' already exists",
        )

    # Create rule in draft status
    try:
        rule = Rule(
            id=request.id,
            field=request.field,
            op=request.op,
            value=request.value,
            action=request.action,
            score=request.score,
            severity=request.severity,
            reason=request.reason,
            status=RuleStatus.DRAFT.value,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid rule: {e}") from e

    # Save to store
    try:
        store.save(rule)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Create version snapshot
    from api.versioning import get_version_store

    version_store = get_version_store()
    version_store.save(
        rule=rule,
        created_by=request.actor,
        reason=f"Created by {request.actor}",
    )

    # Create audit record
    from api.audit import get_audit_logger

    audit_logger = get_audit_logger()
    after_state = {
        "id": rule.id,
        "field": rule.field,
        "op": rule.op,
        "value": rule.value,
        "action": rule.action,
        "score": rule.score,
        "severity": rule.severity,
        "reason": rule.reason,
        "status": rule.status,
    }

    audit_logger.log(
        rule_id=rule.id,
        action="create",
        actor=request.actor,
        before_state=None,
        after_state=after_state,
        reason=f"Rule created by {request.actor}",
    )

    # Run validation against all draft rules and production ruleset
    from api.rules import RuleSet

    draft_rules = store.list_rules(include_archived=False)
    manager = get_model_manager()
    production_ruleset = manager.ruleset

    # Combine draft rules with production rules for validation
    all_rules = draft_rules.copy()
    if production_ruleset:
        # Only include active rules from production
        all_rules.extend([r for r in production_ruleset.rules if r.status == "active"])

    test_ruleset = RuleSet(version="validation", rules=all_rules)
    conflicts, redundancies = validate_ruleset(test_ruleset, strict=False)

    # Convert to response
    rule_response = DraftRuleResponse(
        rule_id=rule.id,
        field=rule.field,
        op=rule.op,
        value=rule.value,
        action=rule.action,
        score=rule.score,
        severity=rule.severity,
        reason=rule.reason,
        status=rule.status,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    validation = ValidationResult(
        conflicts=[
            ConflictResponse(
                rule1_id=c.rule1_id,
                rule2_id=c.rule2_id,
                conflict_type=c.conflict_type,
                description=c.description,
            )
            for c in conflicts
        ],
        redundancies=[
            RedundancyResponse(
                rule_id=r.rule_id,
                redundant_with=r.redundant_with,
                redundancy_type=r.redundancy_type,
                description=r.description,
            )
            for r in redundancies
        ],
        is_valid=len(conflicts) == 0,
    )

    return DraftRuleCreateResponse(
        rule_id=rule.id,
        rule=rule_response,
        validation=validation,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/rules/draft",
    response_model=DraftRuleListResponse,
    tags=["Draft Rules"],
    summary="List all draft rules",
    description="""
List all draft rules with optional status filter.

Returns rules ordered by rule ID. Archived rules are excluded by default.
""",
)
async def list_draft_rules(
    status: str | None = Query(
        None,
        description="Filter by status (draft, archived, etc.)",
    ),
    include_archived: bool = Query(
        False,
        description="Include archived rules in results",
    ),
) -> DraftRuleListResponse:
    """List all draft rules.

    Args:
        status: Optional status filter.
        include_archived: Whether to include archived rules.

    Returns:
        DraftRuleListResponse with list of draft rules.
    """
    from api.draft_store import get_draft_store

    store = get_draft_store()
    rules = store.list_rules(status=status, include_archived=include_archived)

    # Convert to response
    rule_responses = [
        DraftRuleResponse(
            rule_id=rule.id,
            field=rule.field,
            op=rule.op,
            value=rule.value,
            action=rule.action,
            score=rule.score,
            severity=rule.severity,
            reason=rule.reason,
            status=rule.status,
            created_at=None,  # Created_at not tracked in this sub-phase
        )
        for rule in rules
    ]

    return DraftRuleListResponse(rules=rule_responses, total=len(rule_responses))


@app.get(
    "/rules/draft/{rule_id}",
    response_model=DraftRuleResponse,
    tags=["Draft Rules"],
    summary="Get a draft rule by ID",
    description="""
Get a specific draft rule by its identifier.

Returns 404 if rule not found.
""",
)
async def get_draft_rule(rule_id: str) -> DraftRuleResponse:
    """Get a draft rule by ID.

    Args:
        rule_id: Rule identifier.

    Returns:
        DraftRuleResponse with rule details.

    Raises:
        HTTPException: If rule not found.
    """
    from api.draft_store import get_draft_store

    store = get_draft_store()
    rule = store.get(rule_id)

    if rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Draft rule not found: {rule_id}",
        )

    return DraftRuleResponse(
        rule_id=rule.id,
        field=rule.field,
        op=rule.op,
        value=rule.value,
        action=rule.action,
        score=rule.score,
        severity=rule.severity,
        reason=rule.reason,
        status=rule.status,
        created_at=None,  # Created_at not tracked in this sub-phase
    )


@app.put(
    "/rules/draft/{rule_id}",
    response_model=DraftRuleUpdateResponse,
    tags=["Draft Rules"],
    summary="Update a draft rule",
    description="""
Update an existing draft rule.

Only draft rules can be updated. All fields are optional - only provided
fields will be updated. Creates a new version and audit record.
""",
)
async def update_draft_rule(
    rule_id: str, request: DraftRuleUpdateRequest
) -> DraftRuleUpdateResponse:
    """Update a draft rule.

    Args:
        rule_id: Rule identifier.
        request: Update request with optional fields and actor.

    Returns:
        DraftRuleUpdateResponse with updated rule, version ID, and validation.

    Raises:
        HTTPException: If rule not found, not in draft status, or update fails.
    """
    from api.audit import get_audit_logger
    from api.draft_store import get_draft_store
    from api.rules import Rule, RuleSet, RuleStatus
    from api.validation import validate_ruleset
    from api.versioning import get_version_store

    store = get_draft_store()
    version_store = get_version_store()
    audit_logger = get_audit_logger()

    # Get existing rule
    existing_rule = store.get(rule_id)
    if existing_rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Draft rule not found: {rule_id}",
        )

    # Enforce draft-only
    if existing_rule.status != RuleStatus.DRAFT.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update rule {rule_id} with status {existing_rule.status}. "
            "Only draft rules can be updated.",
        )

    # Build updated rule dict
    rule_dict = {
        "id": existing_rule.id,
        "field": request.field if request.field is not None else existing_rule.field,
        "op": request.op if request.op is not None else existing_rule.op,
        "value": request.value if request.value is not None else existing_rule.value,
        "action": (
            request.action if request.action is not None else existing_rule.action
        ),
        "score": request.score if request.score is not None else existing_rule.score,
        "severity": (
            request.severity if request.severity is not None else existing_rule.severity
        ),
        "reason": (
            request.reason if request.reason is not None else existing_rule.reason
        ),
        "status": RuleStatus.DRAFT.value,  # Always keep as draft
    }

    # Create updated rule
    try:
        updated_rule = Rule(**rule_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid rule update: {e}") from e

    # Save before_state for audit
    before_state = {
        "id": existing_rule.id,
        "field": existing_rule.field,
        "op": existing_rule.op,
        "value": existing_rule.value,
        "action": existing_rule.action,
        "score": existing_rule.score,
        "severity": existing_rule.severity,
        "reason": existing_rule.reason,
        "status": existing_rule.status,
    }

    # Save updated rule
    try:
        store.save(updated_rule)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Create version snapshot
    version = version_store.save(
        rule=updated_rule,
        created_by=request.actor,
        reason=f"Updated by {request.actor}",
    )

    # Create audit record
    after_state = {
        "id": updated_rule.id,
        "field": updated_rule.field,
        "op": updated_rule.op,
        "value": updated_rule.value,
        "action": updated_rule.action,
        "score": updated_rule.score,
        "severity": updated_rule.severity,
        "reason": updated_rule.reason,
        "status": updated_rule.status,
    }

    audit_logger.log(
        rule_id=rule_id,
        action="update",
        actor=request.actor,
        before_state=before_state,
        after_state=after_state,
        reason=f"Rule updated by {request.actor}",
    )

    # Run validation
    draft_rules = store.list_rules(include_archived=False)
    manager = get_model_manager()
    production_ruleset = manager.ruleset

    all_rules = draft_rules.copy()
    if production_ruleset:
        all_rules.extend([r for r in production_ruleset.rules if r.status == "active"])

    test_ruleset = RuleSet(version="validation", rules=all_rules)
    conflicts, redundancies = validate_ruleset(test_ruleset, strict=False)

    # Convert to response
    rule_response = DraftRuleResponse(
        rule_id=updated_rule.id,
        field=updated_rule.field,
        op=updated_rule.op,
        value=updated_rule.value,
        action=updated_rule.action,
        score=updated_rule.score,
        severity=updated_rule.severity,
        reason=updated_rule.reason,
        status=updated_rule.status,
        created_at=None,
    )

    validation = ValidationResult(
        conflicts=[
            ConflictResponse(
                rule1_id=c.rule1_id,
                rule2_id=c.rule2_id,
                conflict_type=c.conflict_type,
                description=c.description,
            )
            for c in conflicts
        ],
        redundancies=[
            RedundancyResponse(
                rule_id=r.rule_id,
                redundant_with=r.redundant_with,
                redundancy_type=r.redundancy_type,
                description=r.description,
            )
            for r in redundancies
        ],
        is_valid=len(conflicts) == 0,
    )

    return DraftRuleUpdateResponse(
        rule=rule_response,
        version_id=version.version_id,
        validation=validation,
    )


@app.delete(
    "/rules/draft/{rule_id}",
    tags=["Draft Rules"],
    summary="Archive a draft rule",
    description="""
Archive (soft-delete) a draft rule by changing its status to archived.

Only draft rules can be archived. Creates an audit record for the
state change.
""",
)
async def delete_draft_rule(
    rule_id: str,
    actor: str = Query(..., description="Who is archiving this rule"),
) -> dict:
    """Archive a draft rule.

    Args:
        rule_id: Rule identifier.
        actor: Who is performing the archive.

    Returns:
        Dict with success status and rule ID.

    Raises:
        HTTPException: If rule not found or not in draft status.
    """
    from api.audit import get_audit_logger
    from api.draft_store import get_draft_store
    from api.rules import RuleStatus

    store = get_draft_store()
    audit_logger = get_audit_logger()

    # Get existing rule
    existing_rule = store.get(rule_id)
    if existing_rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Draft rule not found: {rule_id}",
        )

    # Enforce draft-only
    if existing_rule.status != RuleStatus.DRAFT.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot archive rule {rule_id} with status {existing_rule.status}. "
            "Only draft rules can be archived.",
        )

    # Archive the rule
    success = store.delete(rule_id)
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to archive rule {rule_id}",
        )

    # Get archived rule for audit
    archived_rule = store.get(rule_id)

    # Create audit record
    audit_logger.log(
        rule_id=rule_id,
        action="state_change",
        actor=actor,
        before_state={"status": RuleStatus.DRAFT.value},
        after_state={"status": RuleStatus.ARCHIVED.value},
        reason=f"Rule archived by {actor}",
    )

    return {
        "success": True,
        "rule_id": rule_id,
        "status": archived_rule.status if archived_rule else "archived",
    }


@app.post(
    "/rules/draft/{rule_id}/validate",
    response_model=DraftRuleValidateResponse,
    tags=["Draft Rules"],
    summary="Validate a draft rule",
    description="""
Run full validation suite on a draft rule.

Validates schema, checks for conflicts and redundancies. Can optionally
validate against production ruleset or draft-only. Deterministic and
read-only - does not modify rule state.
""",
)
async def validate_draft_rule(
    rule_id: str, request: DraftRuleValidateRequest
) -> DraftRuleValidateResponse:
    """Validate a draft rule.

    Args:
        rule_id: Rule identifier.
        request: Validation request with optional include_existing_rules flag.

    Returns:
        DraftRuleValidateResponse with validation results.

    Raises:
        HTTPException: If rule not found.
    """
    from api.draft_store import get_draft_store
    from api.rules import RuleSet, RuleStatus
    from api.validation import validate_ruleset

    store = get_draft_store()

    # Get the rule
    rule = store.get(rule_id)
    if rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Draft rule not found: {rule_id}",
        )

    # Collect schema errors
    schema_errors = []

    # Validate rule schema (basic checks)
    try:
        # Rule.__post_init__ validates these, but we'll check explicitly
        if rule.op not in [">", ">=", "<", "<=", "==", "in", "not_in"]:
            schema_errors.append(f"Invalid operator: {rule.op}")

        if rule.action not in ["override_score", "clamp_min", "clamp_max", "reject"]:
            schema_errors.append(f"Invalid action: {rule.action}")

        score_actions = ["override_score", "clamp_min", "clamp_max"]
        if rule.action in score_actions and rule.score is None:
            schema_errors.append(f"Action {rule.action} requires 'score' field")

        if rule.op in ["in", "not_in"] and not isinstance(rule.value, list):
            schema_errors.append(f"Operator {rule.op} requires 'value' to be a list")

        if rule.severity not in ["low", "medium", "high"]:
            schema_errors.append(f"Invalid severity: {rule.severity}")

        if rule.status not in [s.value for s in RuleStatus]:
            schema_errors.append(f"Invalid status: {rule.status}")

    except Exception as e:
        schema_errors.append(f"Schema validation error: {e}")

    # Build ruleset for validation
    rules_to_validate = [rule]

    if request.include_existing_rules:
        # Include other draft rules
        other_drafts = [
            r for r in store.list_rules(include_archived=False) if r.id != rule_id
        ]
        rules_to_validate.extend(other_drafts)

        # Include production active rules
        manager = get_model_manager()
        production_ruleset = manager.ruleset
        if production_ruleset:
            rules_to_validate.extend(
                [r for r in production_ruleset.rules if r.status == "active"]
            )

    # Run validation
    test_ruleset = RuleSet(version="validation", rules=rules_to_validate)
    conflicts, redundancies = validate_ruleset(test_ruleset, strict=False)

    # Filter conflicts and redundancies to only those involving this rule
    rule_conflicts = [
        c for c in conflicts if c.rule1_id == rule_id or c.rule2_id == rule_id
    ]
    rule_redundancies = [r for r in redundancies if r.rule_id == rule_id]

    # Convert to response
    return DraftRuleValidateResponse(
        schema_errors=schema_errors,
        conflicts=[
            ConflictResponse(
                rule1_id=c.rule1_id,
                rule2_id=c.rule2_id,
                conflict_type=c.conflict_type,
                description=c.description,
            )
            for c in rule_conflicts
        ],
        redundancies=[
            RedundancyResponse(
                rule_id=r.rule_id,
                redundant_with=r.redundant_with,
                redundancy_type=r.redundancy_type,
                description=r.description,
            )
            for r in rule_redundancies
        ],
        is_valid=len(schema_errors) == 0 and len(rule_conflicts) == 0,
    )


@app.post(
    "/rules/draft/{rule_id}/submit",
    response_model=DraftRuleSubmitResponse,
    tags=["Draft Rules"],
    summary="Submit a draft rule for review",
    description="""
Submit a draft rule for review, transitioning it to pending_review status.

Requires:
- Rule must be valid (no conflicts)
- Justification text (min 10 characters)

Submission is one-way - rule cannot be edited after submission until reviewed.
""",
)
async def submit_draft_rule(
    rule_id: str, request: DraftRuleSubmitRequest
) -> DraftRuleSubmitResponse:
    """Submit a draft rule for review.

    Args:
        rule_id: Rule identifier.
        request: Submit request with actor and justification.

    Returns:
        DraftRuleSubmitResponse with updated rule and submission timestamp.

    Raises:
        HTTPException: If rule not found, not in draft status, validation fails,
            or transition fails.
    """
    from api.draft_store import get_draft_store
    from api.rules import RuleSet, RuleStatus
    from api.validation import validate_ruleset
    from api.versioning import get_version_store
    from api.workflow import RuleStateMachine, TransitionError

    store = get_draft_store()
    version_store = get_version_store()
    state_machine = RuleStateMachine(require_approval=False)

    # Get existing rule
    rule = store.get(rule_id)
    if rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Draft rule not found: {rule_id}",
        )

    # Enforce draft-only
    if rule.status != RuleStatus.DRAFT.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot submit rule {rule_id} with status {rule.status}. "
            "Only draft rules can be submitted.",
        )

    # Run validation - must pass before submission
    draft_rules = store.list_rules(include_archived=False)
    manager = get_model_manager()
    production_ruleset = manager.ruleset

    all_rules = draft_rules.copy()
    if production_ruleset:
        all_rules.extend([r for r in production_ruleset.rules if r.status == "active"])

    test_ruleset = RuleSet(version="validation", rules=all_rules)
    conflicts, redundancies = validate_ruleset(test_ruleset, strict=False)

    # Filter conflicts to only those involving this rule
    rule_conflicts = [
        c for c in conflicts if c.rule1_id == rule_id or c.rule2_id == rule_id
    ]

    # Block submission if conflicts exist
    if rule_conflicts:
        conflict_descriptions = [c.description for c in rule_conflicts]
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot submit rule {rule_id} with conflicts. "
                f"Conflicts: {'; '.join(conflict_descriptions)}"
            ),
        )

    # Transition rule to pending_review
    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.PENDING_REVIEW.value,
            actor=request.actor,
            reason=request.justification,
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Update rule in store (status changed to pending_review)
    # Store now allows updating draft -> pending_review
    try:
        store.save(updated_rule)
    except ValueError as e:
        # Fallback: if store rejects, log warning but continue
        logger.warning(f"Could not update rule in store: {e}")
        store._rules[rule_id] = updated_rule
        store._save_rules()

    # Create version snapshot
    version_store.save(
        rule=updated_rule,
        created_by=request.actor,
        reason=f"Submitted for review: {request.justification}",
    )

    # Audit record is already created by state_machine.transition()

    # Convert to response
    rule_response = DraftRuleResponse(
        rule_id=updated_rule.id,
        field=updated_rule.field,
        op=updated_rule.op,
        value=updated_rule.value,
        action=updated_rule.action,
        score=updated_rule.score,
        severity=updated_rule.severity,
        reason=updated_rule.reason,
        status=updated_rule.status,
        created_at=None,
    )

    return DraftRuleSubmitResponse(
        rule=rule_response,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        audit_id=None,  # Audit ID not available from AuditRecord
    )


@app.post(
    "/rules/draft/{rule_id}/approve",
    response_model=ApproveRuleResponse,
    tags=["Draft Rules"],
    summary="Approve a pending rule",
    description="""
Approve a rule that is pending review, transitioning it to active status.

Requires:
- Rule must be in pending_review status
- Approver must be different from the actor who submitted
- Approval requirement must be enabled
""",
)
async def approve_draft_rule(
    rule_id: str, request: ApproveRuleRequest
) -> ApproveRuleResponse:
    """Approve a draft rule for activation.

    Args:
        rule_id: Rule identifier.
        request: Approve request with approver and optional reason.

    Returns:
        ApproveRuleResponse with approved rule.

    Raises:
        HTTPException: If rule not found, not in pending_review status,
            or transition fails.
    """
    from dataclasses import asdict

    from api.audit import get_audit_logger
    from api.draft_store import get_draft_store
    from api.rules import Rule, RuleStatus
    from api.versioning import get_version_store
    from api.workflow import TransitionError, create_state_machine

    store = get_draft_store()
    version_store = get_version_store()
    audit_logger = get_audit_logger()
    state_machine = create_state_machine()

    # Get existing rule
    rule = store.get(rule_id)
    if rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Draft rule not found: {rule_id}",
        )

    if rule.status != RuleStatus.PENDING_REVIEW.value:
        raise HTTPException(
            status_code=400,
            detail=f"Rule {rule_id} is not pending review (status: {rule.status})",
        )

    # Check for self-approval: find the actor who submitted this rule
    rule_history = audit_logger.get_rule_history(rule_id)
    submitter = None
    for record in reversed(rule_history):
        if (
            record.action == "state_change"
            and record.after_state.get("status") == "pending_review"
        ):
            submitter = record.actor
            break

    if submitter and submitter == request.approver:
        raise HTTPException(
            status_code=400,
            detail=f"Self-approval not allowed. Actor '{request.approver}' cannot "
            "approve their own submission.",
        )

    # Transition to active
    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.ACTIVE.value,
            actor=request.approver,
            reason=request.reason or "Approved for activation",
            approver=request.approver,
            previous_actor=submitter,
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update in store (store allows active status for approved rules)
    rule_dict = asdict(updated_rule)
    rule_dict["status"] = RuleStatus.ACTIVE.value
    active_rule = Rule(**rule_dict)
    # Store active rules separately or update existing
    if rule_id in store._rules:
        store._rules[rule_id] = active_rule
        store._save_rules()

    # Create version snapshot
    version_store.save(
        rule=active_rule,
        created_by=request.approver,
        reason=request.reason or "Approved and activated",
    )

    # Convert to response
    rule_response = DraftRuleResponse(
        rule_id=active_rule.id,
        field=active_rule.field,
        op=active_rule.op,
        value=active_rule.value,
        action=active_rule.action,
        score=active_rule.score,
        severity=active_rule.severity,
        reason=active_rule.reason,
        status=active_rule.status,
        created_at=None,
    )

    return ApproveRuleResponse(
        rule=rule_response,
        approved_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/rules/draft/{rule_id}/reject",
    response_model=RejectRuleResponse,
    tags=["Draft Rules"],
    summary="Reject a pending rule",
    description="""
Reject a rule that is pending review, returning it to draft status.

Requires:
- Rule must be in pending_review status
- Reason for rejection (min 10 characters)
""",
)
async def reject_draft_rule(
    rule_id: str, request: RejectRuleRequest
) -> RejectRuleResponse:
    """Reject a draft rule, returning it to draft status.

    Args:
        rule_id: Rule identifier.
        request: Reject request with actor and reason.

    Returns:
        RejectRuleResponse with rejected rule (back to draft).

    Raises:
        HTTPException: If rule not found, not in pending_review status,
            or transition fails.
    """
    from api.draft_store import get_draft_store
    from api.rules import RuleStatus
    from api.versioning import get_version_store
    from api.workflow import TransitionError, create_state_machine

    store = get_draft_store()
    version_store = get_version_store()
    state_machine = create_state_machine()

    # Get existing rule
    rule = store.get(rule_id)
    if rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Draft rule not found: {rule_id}",
        )

    if rule.status != RuleStatus.PENDING_REVIEW.value:
        raise HTTPException(
            status_code=400,
            detail=f"Rule {rule_id} is not pending review (status: {rule.status})",
        )

    # Transition back to draft
    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.DRAFT.value,
            actor=request.actor,
            reason=request.reason,
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update in store
    store.save(updated_rule)

    # Create version snapshot
    version_store.save(
        rule=updated_rule,
        created_by=request.actor,
        reason=f"Rejected: {request.reason}",
    )

    # Convert to response
    rule_response = DraftRuleResponse(
        rule_id=updated_rule.id,
        field=updated_rule.field,
        op=updated_rule.op,
        value=updated_rule.value,
        action=updated_rule.action,
        score=updated_rule.score,
        severity=updated_rule.severity,
        reason=updated_rule.reason,
        status=updated_rule.status,
        created_at=None,
    )

    return RejectRuleResponse(
        rule=rule_response,
        rejected_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/rules/{rule_id}/activate",
    response_model=ActivateRuleResponse,
    tags=["Rules"],
    summary="Activate a rule",
    description="""
Activate a rule, transitioning it to active status.

Can activate from:
- pending_review (requires approval)
- shadow (requires approval)
- disabled (requires approval)

Requires:
- Reason for activation (min 10 characters)
- Approver if transition requires approval
""",
)
async def activate_rule(
    rule_id: str, request: ActivateRuleRequest
) -> ActivateRuleResponse:
    """Activate a rule.

    Args:
        rule_id: Rule identifier.
        request: Activate request with actor, optional approver, and reason.

    Returns:
        ActivateRuleResponse with activated rule.

    Raises:
        HTTPException: If rule not found, invalid status, or transition fails.
    """
    from dataclasses import asdict

    from api.draft_store import get_draft_store
    from api.model_manager import get_model_manager
    from api.rules import Rule, RuleStatus
    from api.versioning import get_version_store
    from api.workflow import TransitionError, create_state_machine

    store = get_draft_store()
    version_store = get_version_store()
    state_machine = create_state_machine()

    # Try to get from draft store first
    rule = store.get(rule_id)
    if rule is None:
        # Try to get from production ruleset
        manager = get_model_manager()
        if manager.ruleset:
            for r in manager.ruleset.rules:
                if r.id == rule_id:
                    rule = r
                    break

    if rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Rule not found: {rule_id}",
        )

    # Check if transition is allowed
    if rule.status not in [
        RuleStatus.PENDING_REVIEW.value,
        RuleStatus.SHADOW.value,
        RuleStatus.DISABLED.value,
    ]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot activate rule {rule_id} from status {rule.status}",
        )

    # Transition to active
    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.ACTIVE.value,
            actor=request.actor,
            reason=request.reason,
            approver=request.approver,
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update in store if it exists there
    if rule_id in store._rules:
        rule_dict = asdict(updated_rule)
        rule_dict["status"] = RuleStatus.ACTIVE.value
        active_rule = Rule(**rule_dict)
        store._rules[rule_id] = active_rule
        store._save_rules()

    # Create version snapshot
    version_store.save(
        rule=updated_rule,
        created_by=request.actor,
        reason=request.reason,
    )

    # Convert to response
    rule_response = DraftRuleResponse(
        rule_id=updated_rule.id,
        field=updated_rule.field,
        op=updated_rule.op,
        value=updated_rule.value,
        action=updated_rule.action,
        score=updated_rule.score,
        severity=updated_rule.severity,
        reason=updated_rule.reason,
        status=updated_rule.status,
        created_at=None,
    )

    return ActivateRuleResponse(
        rule=rule_response,
        activated_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/rules/{rule_id}/disable",
    response_model=DisableRuleResponse,
    tags=["Rules"],
    summary="Disable a rule",
    description="""
Disable a rule, transitioning it to disabled status.

Can disable from:
- active
- shadow

Requires:
- Optional reason for disabling
""",
)
async def disable_rule(
    rule_id: str, request: DisableRuleRequest
) -> DisableRuleResponse:
    """Disable a rule.

    Args:
        rule_id: Rule identifier.
        request: Disable request with actor and optional reason.

    Returns:
        DisableRuleResponse with disabled rule.

    Raises:
        HTTPException: If rule not found, invalid status, or transition fails.
    """
    from dataclasses import asdict

    from api.draft_store import get_draft_store
    from api.model_manager import get_model_manager
    from api.rules import Rule, RuleStatus
    from api.versioning import get_version_store
    from api.workflow import TransitionError, create_state_machine

    store = get_draft_store()
    version_store = get_version_store()
    state_machine = create_state_machine()

    # Try to get from draft store first
    rule = store.get(rule_id)
    if rule is None:
        # Try to get from production ruleset
        manager = get_model_manager()
        if manager.ruleset:
            for r in manager.ruleset.rules:
                if r.id == rule_id:
                    rule = r
                    break

    if rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Rule not found: {rule_id}",
        )

    # Check if transition is allowed
    if rule.status not in [RuleStatus.ACTIVE.value, RuleStatus.SHADOW.value]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot disable rule {rule_id} from status {rule.status}",
        )

    # Transition to disabled
    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.DISABLED.value,
            actor=request.actor,
            reason=request.reason or f"Disabled by {request.actor}",
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update in store if it exists there
    if rule_id in store._rules:
        rule_dict = asdict(updated_rule)
        rule_dict["status"] = RuleStatus.DISABLED.value
        disabled_rule = Rule(**rule_dict)
        store._rules[rule_id] = disabled_rule
        store._save_rules()

    # Create version snapshot
    version_store.save(
        rule=updated_rule,
        created_by=request.actor,
        reason=request.reason or "Disabled",
    )

    # Convert to response
    rule_response = DraftRuleResponse(
        rule_id=updated_rule.id,
        field=updated_rule.field,
        op=updated_rule.op,
        value=updated_rule.value,
        action=updated_rule.action,
        score=updated_rule.score,
        severity=updated_rule.severity,
        reason=updated_rule.reason,
        status=updated_rule.status,
        created_at=None,
    )

    return DisableRuleResponse(
        rule=rule_response,
        disabled_at=datetime.now(timezone.utc).isoformat(),
    )


@app.post(
    "/rules/{rule_id}/shadow",
    response_model=ShadowRuleResponse,
    tags=["Rules"],
    summary="Move a rule to shadow mode",
    description="""
Move a rule to shadow mode, transitioning it from active to shadow status.

Shadow rules are evaluated but do not affect scores.

Can shadow from:
- active

Requires:
- Optional reason for shadow mode
""",
)
async def shadow_rule(rule_id: str, request: ShadowRuleRequest) -> ShadowRuleResponse:
    """Move a rule to shadow mode.

    Args:
        rule_id: Rule identifier.
        request: Shadow request with actor and optional reason.

    Returns:
        ShadowRuleResponse with shadowed rule.

    Raises:
        HTTPException: If rule not found, invalid status, or transition fails.
    """
    from dataclasses import asdict

    from api.draft_store import get_draft_store
    from api.model_manager import get_model_manager
    from api.rules import Rule, RuleStatus
    from api.versioning import get_version_store
    from api.workflow import TransitionError, create_state_machine

    store = get_draft_store()
    version_store = get_version_store()
    state_machine = create_state_machine()

    # Try to get from draft store first
    rule = store.get(rule_id)
    if rule is None:
        # Try to get from production ruleset
        manager = get_model_manager()
        if manager.ruleset:
            for r in manager.ruleset.rules:
                if r.id == rule_id:
                    rule = r
                    break

    if rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"Rule not found: {rule_id}",
        )

    # Check if transition is allowed
    if rule.status != RuleStatus.ACTIVE.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot shadow rule {rule_id} from status {rule.status}",
        )

    # Transition to shadow
    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.SHADOW.value,
            actor=request.actor,
            reason=request.reason or f"Moved to shadow by {request.actor}",
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update in store if it exists there
    if rule_id in store._rules:
        rule_dict = asdict(updated_rule)
        rule_dict["status"] = RuleStatus.SHADOW.value
        shadow_rule_obj = Rule(**rule_dict)
        store._rules[rule_id] = shadow_rule_obj
        store._save_rules()

    # Create version snapshot
    version_store.save(
        rule=updated_rule,
        created_by=request.actor,
        reason=request.reason or "Moved to shadow",
    )

    # Convert to response
    rule_response = DraftRuleResponse(
        rule_id=updated_rule.id,
        field=updated_rule.field,
        op=updated_rule.op,
        value=updated_rule.value,
        action=updated_rule.action,
        score=updated_rule.score,
        severity=updated_rule.severity,
        reason=updated_rule.reason,
        status=updated_rule.status,
        created_at=None,
    )

    return ShadowRuleResponse(
        rule=rule_response,
        shadowed_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/rules/{rule_id}/versions",
    response_model=RuleVersionListResponse,
    tags=["Rules"],
    summary="List all versions of a rule",
    description="Get all versions of a rule, ordered by timestamp (oldest first).",
)
async def list_rule_versions(rule_id: str) -> RuleVersionListResponse:
    """List all versions of a rule.

    Args:
        rule_id: Rule identifier.

    Returns:
        RuleVersionListResponse with list of versions.
    """
    from api.versioning import get_version_store

    version_store = get_version_store()
    versions = version_store.list_versions(rule_id)

    version_responses = [
        RuleVersionResponse(
            rule_id=v.rule_id,
            version_id=v.version_id,
            rule=DraftRuleResponse(
                rule_id=v.rule.id,
                field=v.rule.field,
                op=v.rule.op,
                value=v.rule.value,
                action=v.rule.action,
                score=v.rule.score,
                severity=v.rule.severity,
                reason=v.rule.reason,
                status=v.rule.status,
                created_at=None,
            ),
            timestamp=v.timestamp.isoformat(),
            created_by=v.created_by,
            reason=v.reason,
        )
        for v in versions
    ]

    return RuleVersionListResponse(
        versions=version_responses, total=len(version_responses)
    )


@app.get(
    "/rules/{rule_id}/versions/{version_id}",
    response_model=RuleVersionResponse,
    tags=["Rules"],
    summary="Get a specific version of a rule",
    description="Get details of a specific version of a rule.",
)
async def get_rule_version(rule_id: str, version_id: str) -> RuleVersionResponse:
    """Get a specific version of a rule.

    Args:
        rule_id: Rule identifier.
        version_id: Version identifier.

    Returns:
        RuleVersionResponse with version details.

    Raises:
        HTTPException: If version not found.
    """
    from api.versioning import get_version_store

    version_store = get_version_store()
    version = version_store.get_version(rule_id, version_id)

    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version_id} not found for rule {rule_id}",
        )

    return RuleVersionResponse(
        rule_id=version.rule_id,
        version_id=version.version_id,
        rule=DraftRuleResponse(
            rule_id=version.rule.id,
            field=version.rule.field,
            op=version.rule.op,
            value=version.rule.value,
            action=version.rule.action,
            score=version.rule.score,
            severity=version.rule.severity,
            reason=version.rule.reason,
            status=version.rule.status,
            created_at=None,
        ),
        timestamp=version.timestamp.isoformat(),
        created_by=version.created_by,
        reason=version.reason,
    )


@app.post(
    "/rules/{rule_id}/versions/{version_id}/rollback",
    response_model=RollbackRuleResponse,
    tags=["Rules"],
    summary="Rollback a rule to a previous version",
    description="""
Rollback a rule to a previous version.

This creates a new version (does not delete history).

Requires:
- Rule must exist
- Version must exist
- Optional reason for rollback
""",
)
async def rollback_rule_version(
    rule_id: str, version_id: str, request: RollbackRuleRequest
) -> RollbackRuleResponse:
    """Rollback a rule to a previous version.

    Args:
        rule_id: Rule identifier.
        version_id: Version to rollback to.
        request: Rollback request with actor and optional reason.

    Returns:
        RollbackRuleResponse with rolled back rule.

    Raises:
        HTTPException: If rule or version not found, or rollback fails.
    """
    from api.draft_store import get_draft_store
    from api.versioning import get_version_store

    store = get_draft_store()
    version_store = get_version_store()

    # Verify version exists
    version = version_store.get_version(rule_id, version_id)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version_id} not found for rule {rule_id}",
        )

    # Perform rollback
    try:
        new_version = version_store.rollback(
            rule_id=rule_id,
            version_id=version_id,
            rolled_back_by=request.actor,
            reason=request.reason,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update rule in store if it exists
    rule = store.get(rule_id)
    if rule is not None:
        # Update to rolled back version
        store.save(new_version.rule)

    # Convert to response
    rule_response = DraftRuleResponse(
        rule_id=new_version.rule.id,
        field=new_version.rule.field,
        op=new_version.rule.op,
        value=new_version.rule.value,
        action=new_version.rule.action,
        score=new_version.rule.score,
        severity=new_version.rule.severity,
        reason=new_version.rule.reason,
        status=new_version.rule.status,
        created_at=None,
    )

    return RollbackRuleResponse(
        rule=rule_response,
        version_id=new_version.version_id,
        rolled_back_to=version_id,
        rolled_back_at=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/audit/logs",
    response_model=AuditLogQueryResponse,
    tags=["Audit"],
    summary="Query audit logs",
    description="""
Query audit logs with optional filters.

Filters:
- rule_id: Filter by rule ID
- actor: Filter by actor
- action: Filter by action type
- start_date: Filter records after this date (ISO format)
- end_date: Filter records before this date (ISO format)
""",
)
async def query_audit_logs(
    rule_id: str | None = Query(None, description="Filter by rule ID"),
    actor: str | None = Query(None, description="Filter by actor"),
    action: str | None = Query(None, description="Filter by action type"),
    start_date: str | None = Query(None, description="Start date (ISO format)"),
    end_date: str | None = Query(None, description="End date (ISO format)"),
) -> AuditLogQueryResponse:
    """Query audit logs with filters.

    Args:
        rule_id: Optional rule ID filter.
        actor: Optional actor filter.
        action: Optional action type filter.
        start_date: Optional start date (ISO format string).
        end_date: Optional end date (ISO format string).

    Returns:
        AuditLogQueryResponse with matching records.
    """
    from api.audit import get_audit_logger

    audit_logger = get_audit_logger()

    # Parse dates if provided
    start_dt = None
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid start_date format: {start_date}"
            )

    end_dt = None
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid end_date format: {end_date}"
            )

    # Query audit logs
    records = audit_logger.query(
        rule_id=rule_id,
        actor=actor,
        action=action,
        start_date=start_dt,
        end_date=end_dt,
    )

    # Convert to response
    record_responses = [
        AuditRecordResponse(
            rule_id=r.rule_id,
            action=r.action,
            actor=r.actor,
            timestamp=r.timestamp.isoformat(),
            before_state=r.before_state,
            after_state=r.after_state,
            reason=r.reason,
        )
        for r in records
    ]

    return AuditLogQueryResponse(records=record_responses, total=len(record_responses))


@app.get(
    "/audit/rules/{rule_id}/history",
    response_model=AuditLogQueryResponse,
    tags=["Audit"],
    summary="Get audit history for a rule",
    description="Get complete audit history for a specific rule, ordered by timestamp.",
)
async def get_rule_audit_history(rule_id: str) -> AuditLogQueryResponse:
    """Get audit history for a rule.

    Args:
        rule_id: Rule identifier.

    Returns:
        AuditLogQueryResponse with all audit records for the rule.
    """
    from api.audit import get_audit_logger

    audit_logger = get_audit_logger()
    records = audit_logger.get_rule_history(rule_id)

    # Convert to response
    record_responses = [
        AuditRecordResponse(
            rule_id=r.rule_id,
            action=r.action,
            actor=r.actor,
            timestamp=r.timestamp.isoformat(),
            before_state=r.before_state,
            after_state=r.after_state,
            reason=r.reason,
        )
        for r in records
    ]

    return AuditLogQueryResponse(records=record_responses, total=len(record_responses))


@app.get(
    "/audit/export",
    tags=["Audit"],
    summary="Export audit logs",
    description="""
Export audit logs in JSON or CSV format.

Query parameters:
- format: json or csv (default: json)
- rule_id: Optional filter by rule ID
- actor: Optional filter by actor
- action: Optional filter by action type
- start_date: Optional start date (ISO format)
- end_date: Optional end date (ISO format)
""",
)
async def export_audit_logs(
    format: str = Query("json", description="Export format: json or csv"),
    rule_id: str | None = Query(None, description="Filter by rule ID"),
    actor: str | None = Query(None, description="Filter by actor"),
    action: str | None = Query(None, description="Filter by action type"),
    start_date: str | None = Query(None, description="Start date (ISO format)"),
    end_date: str | None = Query(None, description="End date (ISO format)"),
):
    """Export audit logs.

    Args:
        format: Export format (json or csv).
        rule_id: Optional rule ID filter.
        actor: Optional actor filter.
        action: Optional action type filter.
        start_date: Optional start date (ISO format string).
        end_date: Optional end date (ISO format string).

    Returns:
        JSON or CSV response with audit records.
    """
    import csv
    import io

    from fastapi.responses import Response

    from api.audit import get_audit_logger

    if format not in ["json", "csv"]:
        raise HTTPException(
            status_code=400, detail=f"Invalid format: {format}. Use 'json' or 'csv'"
        )

    audit_logger = get_audit_logger()

    # Parse dates if provided
    start_dt = None
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid start_date format: {start_date}"
            )

    end_dt = None
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"Invalid end_date format: {end_date}"
            )

    # Query audit logs
    records = audit_logger.query(
        rule_id=rule_id,
        actor=actor,
        action=action,
        start_date=start_dt,
        end_date=end_dt,
    )

    if format == "json":
        import json

        records_dict = [
            {
                "rule_id": r.rule_id,
                "action": r.action,
                "actor": r.actor,
                "timestamp": r.timestamp.isoformat(),
                "before_state": r.before_state,
                "after_state": r.after_state,
                "reason": r.reason,
            }
            for r in records
        ]
        return Response(
            content=json.dumps(records_dict, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=audit_logs.json"},
        )
    else:  # CSV
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "rule_id",
                "action",
                "actor",
                "timestamp",
                "before_state",
                "after_state",
                "reason",
            ]
        )

        for r in records:
            writer.writerow(
                [
                    r.rule_id,
                    r.action,
                    r.actor,
                    r.timestamp.isoformat(),
                    str(r.before_state) if r.before_state else "",
                    str(r.after_state) if r.after_state else "",
                    r.reason,
                ]
            )

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=audit_logs.csv"},
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
