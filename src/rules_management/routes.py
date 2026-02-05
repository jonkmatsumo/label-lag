import logging
import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query
from google.protobuf.json_format import MessageToDict

from api.audit import get_audit_logger
from api.readiness import CheckStatus, ReadinessEvaluator
from api.crud_client import get_crud_client
from api.errors import analytics_http_exception
from api.schemas import (
    AcceptSuggestionRequest,
    AcceptSuggestionResponse,
    ActivateRuleRequest,
    ActivateRuleResponse,
    ApprovalSignalsResponse,
    ApproveRuleRequest,
    ApproveRuleResponse,
    AuditLogQueryResponse,
    AuditRecordResponse,
    BacktestComparisonResult,
    BacktestMetricsResponse,
    BacktestResultResponse,
    BacktestResultsListResponse,
    BacktestRunRequest,
    CompareRulesetsRequest,
    ConflictResponse,
    CorrelationPair,
    DatasetCorrelationsResponse,
    DatasetRelationshipsResponse,
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
    PublishRuleRequest,
    PublishRuleResponse,
    RedundancyResponse,
    RejectRuleRequest,
    RejectRuleResponse,
    RelationshipMetric,
    RollbackRuleRequest,
    RollbackRuleResponse,
    RuleAnalyticsResponse,
    RuleAttributionResponse,
    RuleDiffResponse,
    RuleFieldChangeResponse,
    RuleHealthResponse,
    RuleMetricsItem,
    RuleSuggestionResponse,
    RuleVersionListResponse,
    RuleVersionResponse,
    SandboxDiffRequest,
    SandboxDiffResponse,
    SandboxEvaluateRequest,
    SandboxEvaluateResponse,
    SandboxMatchedRule,
    ShadowComparisonResponse,
    ShadowRuleRequest,
    ShadowRuleResponse,
    SuggestionEvidence,
    SuggestionsListResponse,
    ValidationResult,
)
from forecast.model_manager import get_model_manager
from rules_management.backtest import BacktestRunner, BacktestStore, BacktestComparator
from rules_management.draft_store import get_draft_store
from rules_management.metrics import get_metrics_collector
from rules_management.rules import Rule, RuleSet, RuleStatus
from rules_management.versioning import get_version_store, diff_rule_versions
from rules_management.workflow import RuleStateMachine, TransitionError, create_state_machine
from rules_management.analytics import RuleHealthEvaluator
from rules_management.attribution import AttributionService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/rules",
    tags=["Rule Inspector"],
    summary="List current production ruleset",
)
async def get_rules() -> dict:
    """Get the current production ruleset."""
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


@router.post(
    "/rules/sandbox/evaluate",
    response_model=SandboxEvaluateResponse,
    tags=["Rule Inspector"],
    summary="Evaluate rules in sandbox mode",
)
async def sandbox_evaluate(request: SandboxEvaluateRequest) -> SandboxEvaluateResponse:
    """Evaluate rules against features in sandbox mode."""
    from api.gateway_client import get_gateway_client

    features = {
        "velocity_24h": request.features.velocity_24h,
        "amount_to_avg_ratio_30d": request.features.amount_to_avg_ratio_30d,
        "balance_volatility_z_score": request.features.balance_volatility_z_score,
        "bank_connections_24h": request.features.bank_connections_24h,
        "merchant_risk_score": request.features.merchant_risk_score,
        "has_history": request.features.has_history,
        "transaction_amount": request.features.transaction_amount,
    }

    ruleset_dict = None
    if request.ruleset is not None:
        ruleset_dict = request.ruleset.model_dump()

    client = get_gateway_client()
    try:
        result = client.evaluate_rules(
            features=features,
            base_score=request.base_score,
            ruleset=ruleset_dict,
            shadow_mode=request.shadow_mode,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Sandbox evaluation failed via gateway")
        raise HTTPException(status_code=500, detail=str(e)) from e

    matched_rules = [
        SandboxMatchedRule(
            rule_id=exp["rule_id"],
            severity=exp["severity"],
            reason=exp["reason"],
            action=exp.get("action", ""),
            score=exp.get("score"),
        )
        for exp in result.get("explanations", [])
    ]

    shadow_matched_rules = [
        SandboxMatchedRule(
            rule_id=exp["rule_id"],
            severity=exp["severity"],
            reason=exp["reason"],
            action=exp.get("action", ""),
            score=exp.get("score"),
        )
        for exp in result.get("shadow_explanations", [])
    ]

    return SandboxEvaluateResponse(
        final_score=result["final_score"],
        baseline_score=result.get("baseline_score", request.base_score),
        shadow_score=result.get("shadow_score"),
        matched_rules=matched_rules,
        explanations=result.get("explanations", []),
        shadow_matched_rules=shadow_matched_rules,
        rejected=result.get("rejected", False),
        ruleset_version=result.get("ruleset_version", "unknown"),
    )


@router.post(
    "/rules/sandbox/diff",
    response_model=SandboxDiffResponse,
    tags=["Rule Inspector"],
    summary="Compare two rulesets in sandbox mode",
)
async def sandbox_diff(request: SandboxDiffRequest) -> SandboxDiffResponse:
    """Compare two rulesets against features in sandbox mode."""
    from api.gateway_client import get_gateway_client

    features = {
        "velocity_24h": request.features.velocity_24h,
        "amount_to_avg_ratio_30d": request.features.amount_to_avg_ratio_30d,
        "balance_volatility_z_score": request.features.balance_volatility_z_score,
        "bank_connections_24h": request.features.bank_connections_24h,
        "merchant_risk_score": request.features.merchant_risk_score,
        "has_history": request.features.has_history,
        "transaction_amount": request.features.transaction_amount,
    }

    ruleset_a_dict = None
    if request.ruleset_a is not None:
        ruleset_a_dict = request.ruleset_a.model_dump()

    ruleset_b_dict = None
    if request.ruleset_b is not None:
        ruleset_b_dict = request.ruleset_b.model_dump()

    client = get_gateway_client()
    try:
        result = client.diff_rules(
            features=features,
            base_score=request.base_score,
            ruleset_a=ruleset_a_dict,
            ruleset_b=ruleset_b_dict,
            shadow_mode=request.shadow_mode,
        )
    except Exception as e:
        logger.exception("Sandbox diff failed via gateway")
        raise HTTPException(status_code=500, detail=str(e)) from e

    def map_eval(data):
        m_rules = [
            SandboxMatchedRule(
                rule_id=exp["rule_id"],
                severity=exp["severity"],
                reason=exp["reason"],
                action=exp.get("action", ""),
                score=exp.get("score"),
            )
            for exp in data.get("explanations", [])
        ]
        s_rules = [
            SandboxMatchedRule(
                rule_id=exp["rule_id"],
                severity=exp["severity"],
                reason=exp["reason"],
                action=exp.get("action", ""),
                score=exp.get("score"),
            )
            for exp in data.get("shadow_explanations", [])
        ]
        return SandboxEvaluateResponse(
            final_score=data["final_score"],
            baseline_score=data.get("baseline_score", request.base_score),
            shadow_score=data.get("shadow_score"),
            matched_rules=m_rules,
            explanations=data.get("explanations", []),
            shadow_matched_rules=s_rules,
            rejected=data.get("rejected", False),
            ruleset_version=data.get("ruleset_version", "unknown"),
        )

    return SandboxDiffResponse(
        a=map_eval(result["a"]),
        b=map_eval(result["b"]),
        diff=result["diff"],
    )


@router.get(
    "/metrics/shadow/comparison",
    response_model=ShadowComparisonResponse,
    tags=["Rule Inspector"],
    summary="Get shadow mode comparison metrics",
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
    """Get shadow mode comparison metrics."""
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format. Use ISO format (YYYY-MM-DD): {e}",
        ) from e

    if rule_ids:
        rule_id_list = [r.strip() for r in rule_ids.split(",") if r.strip()]
    else:
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

    collector = get_metrics_collector()
    report = collector.generate_comparison_report(rule_id_list, start_dt, end_dt)

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


@router.get(
    "/backtest/results",
    response_model=BacktestResultsListResponse,
    tags=["Rule Inspector"],
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
    """List backtest results with optional filters."""
    store = BacktestStore()
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

    results = store.list_results(
        rule_id=rule_id,
        start_date=start_dt,
        end_date=end_dt,
    )
    results = results[:limit]

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


@router.get(
    "/backtest/results/{job_id}",
    response_model=BacktestResultResponse,
    tags=["Rule Inspector"],
)
async def get_backtest_result(job_id: str) -> BacktestResultResponse:
    """Get a specific backtest result by job ID."""
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


@router.get(
    "/suggestions/heuristic",
    response_model=SuggestionsListResponse,
    tags=["Rule Inspector"],
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
    """Get heuristic rule suggestions."""
    from rules_management.suggestions import SuggestionEngine

    try:
        engine = SuggestionEngine(min_confidence=min_confidence)
        suggestions = engine.generate_suggestions(field=field, min_samples=min_samples)

        fingerprint = None
        try:
            client = get_crud_client()
            fp_resp = client.get_dataset_fingerprint()
            if fp_resp:
                import hashlib
                s = f"{fp_resp.generated_records.count}-{fp_resp.feature_snapshots.count}"
                fingerprint = hashlib.sha256(s.encode()).hexdigest()
        except Exception:
            pass

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
                    dataset_fingerprint=fingerprint,
                )
            )

        return SuggestionsListResponse(
            suggestions=response_suggestions,
            total=len(response_suggestions),
        )

    except Exception as e:
        logger.warning(f"Suggestion generation failed: {e}")
        return SuggestionsListResponse(suggestions=[], total=0)


@router.post(
    "/suggestions/accept",
    response_model=AcceptSuggestionResponse,
    tags=["Rule Inspector"],
)
async def accept_suggestion(
    request: AcceptSuggestionRequest,
) -> AcceptSuggestionResponse:
    """Accept a suggestion and create a draft rule."""
    from rules_management.suggestions import RuleSuggestion

    store = get_draft_store()
    version_store = get_version_store()
    audit_logger = get_audit_logger()

    suggestion_data = request.suggestion
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

    if request.edits:
        for key, val in request.edits.items():
            if hasattr(suggestion, key):
                setattr(suggestion, key, val)

    rule_id = request.custom_id
    rule = suggestion.to_rule(rule_id=rule_id)

    if store.exists(rule.id):
        raise HTTPException(
            status_code=409,
            detail=f"Rule with ID '{rule.id}' already exists",
        )

    try:
        store.save(rule)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    version_store.save(
        rule=rule,
        created_by=request.actor,
        reason=f"Accepted suggestion with confidence {suggestion.confidence:.2f}",
    )

    audit_logger.log(
        rule_id=rule.id,
        action="create",
        actor=request.actor,
        before_state=None,
        after_state={
            "id": rule.id,
            "field": rule.field,
            "op": rule.op,
            "value": rule.value,
            "action": rule.action,
            "score": rule.score,
            "severity": rule.severity,
            "reason": rule.reason,
            "status": rule.status,
        },
        reason=f"Accepted suggestion: {suggestion.field} {suggestion.operator} {suggestion.threshold}",
    )

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

    return AcceptSuggestionResponse(
        rule=rule_response,
        rule_id=rule.id,
        source_suggestion={
            "confidence": suggestion.confidence,
            "evidence": evidence_dict,
            "field": suggestion.field,
            "threshold": suggestion.threshold,
        },
    )


@router.post(
    "/rules/draft",
    response_model=DraftRuleCreateResponse,
    tags=["Draft Rules"],
)
async def create_draft_rule(request: DraftRuleCreateRequest) -> DraftRuleCreateResponse:
    """Create a new draft rule."""
    store = get_draft_store()

    if store.exists(request.id):
        raise HTTPException(
            status_code=409,
            detail=f"Rule with ID '{request.id}' already exists",
        )

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

    try:
        store.save(rule)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    version_store = get_version_store()
    version_store.save(
        rule=rule,
        created_by=request.actor,
        reason=f"Created by {request.actor}",
    )

    audit_logger = get_audit_logger()
    audit_logger.log(
        rule_id=rule.id,
        action="create",
        actor=request.actor,
        before_state=None,
        after_state={
            "id": rule.id,
            "field": rule.field,
            "op": rule.op,
            "value": rule.value,
            "action": rule.action,
            "score": rule.score,
            "severity": rule.severity,
            "reason": rule.reason,
            "status": rule.status,
        },
        reason=f"Rule created by {request.actor}",
    )

    draft_rules = store.list_rules(include_archived=False)
    manager = get_model_manager()
    production_ruleset = manager.ruleset

    all_rules = draft_rules.copy()
    if production_ruleset:
        all_rules.extend([r for r in production_ruleset.rules if r.status == "active"])

    test_ruleset = RuleSet(version="validation", rules=all_rules)
    from rules_management.validation import validate_ruleset
    conflicts, redundancies = validate_ruleset(test_ruleset, strict=False)

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


@router.get(
    "/rules/draft",
    response_model=DraftRuleListResponse,
    tags=["Draft Rules"],
)
async def list_draft_rules(
    status: str | None = Query(None, description="Filter by status"),
    include_archived: bool = Query(False, description="Include archived rules"),
) -> DraftRuleListResponse:
    """List all draft rules."""
    store = get_draft_store()
    rules = store.list_rules(status=status, include_archived=include_archived)

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
            created_at=None,
        )
        for rule in rules
    ]

    return DraftRuleListResponse(rules=rule_responses, total=len(rule_responses))


@router.get(
    "/rules/draft/{rule_id}",
    response_model=DraftRuleResponse,
    tags=["Draft Rules"],
)
async def get_draft_rule(rule_id: str) -> DraftRuleResponse:
    """Get a draft rule by ID."""
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
        created_at=None,
    )


@router.put(
    "/rules/draft/{rule_id}",
    response_model=DraftRuleUpdateResponse,
    tags=["Draft Rules"],
)
async def update_draft_rule(
    rule_id: str, request: DraftRuleUpdateRequest
) -> DraftRuleUpdateResponse:
    """Update a draft rule."""
    store = get_draft_store()
    version_store = get_version_store()
    audit_logger = get_audit_logger()

    existing_rule = store.get(rule_id)
    if existing_rule is None:
        raise HTTPException(status_code=404, detail=f"Draft rule not found: {rule_id}")

    if existing_rule.status != RuleStatus.DRAFT.value:
        raise HTTPException(
            status_code=400,
            detail="Only draft rules can be updated.",
        )

    rule_dict = {
        "id": existing_rule.id,
        "field": request.field if request.field is not None else existing_rule.field,
        "op": request.op if request.op is not None else existing_rule.op,
        "value": request.value if request.value is not None else existing_rule.value,
        "action": request.action if request.action is not None else existing_rule.action,
        "score": request.score if request.score is not None else existing_rule.score,
        "severity": request.severity if request.severity is not None else existing_rule.severity,
        "reason": request.reason if request.reason is not None else existing_rule.reason,
        "status": RuleStatus.DRAFT.value,
    }

    try:
        updated_rule = Rule(**rule_dict)
        store.save(updated_rule)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid rule update: {e}") from e

    version = version_store.save(
        rule=updated_rule,
        created_by=request.actor,
        reason=f"Updated by {request.actor}",
    )

    audit_logger.log(
        rule_id=rule_id,
        action="update",
        actor=request.actor,
        before_state=existing_rule.__dict__,
        after_state=updated_rule.__dict__,
        reason=f"Rule updated by {request.actor}",
    )

    from rules_management.validation import validate_ruleset
    draft_rules = store.list_rules(include_archived=False)
    manager = get_model_manager()
    all_rules = draft_rules + ([r for r in manager.ruleset.rules if r.status == "active"] if manager.ruleset else [])
    test_ruleset = RuleSet(version="validation", rules=all_rules)
    conflicts, redundancies = validate_ruleset(test_ruleset, strict=False)

    return DraftRuleUpdateResponse(
        rule=DraftRuleResponse(
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
        ),
        version_id=version.version_id,
        validation=ValidationResult(
            conflicts=[ConflictResponse(rule1_id=c.rule1_id, rule2_id=c.rule2_id, conflict_type=c.conflict_type, description=c.description) for c in conflicts],
            redundancies=[RedundancyResponse(rule_id=r.rule_id, redundant_with=r.redundant_with, redundancy_type=r.redundancy_type, description=r.description) for r in redundancies],
            is_valid=len(conflicts) == 0,
        ),
    )


@router.delete(
    "/rules/draft/{rule_id}",
    tags=["Draft Rules"],
)
async def delete_draft_rule(
    rule_id: str,
    actor: str = Query(..., description="Who is archiving this rule"),
) -> dict:
    """Archive a draft rule."""
    store = get_draft_store()
    audit_logger = get_audit_logger()

    existing_rule = store.get(rule_id)
    if existing_rule is None:
        raise HTTPException(status_code=404, detail=f"Draft rule not found: {rule_id}")

    if existing_rule.status != RuleStatus.DRAFT.value:
        raise HTTPException(status_code=400, detail="Only draft rules can be archived.")

    if store.delete(rule_id):
        audit_logger.log(
            rule_id=rule_id,
            action="state_change",
            actor=actor,
            before_state={"status": RuleStatus.DRAFT.value},
            after_state={"status": RuleStatus.ARCHIVED.value},
            reason=f"Rule archived by {actor}",
        )
        return {"success": True, "rule_id": rule_id, "status": "archived"}
    
    raise HTTPException(status_code=500, detail="Failed to archive rule")


@router.post(
    "/rules/draft/{rule_id}/validate",
    response_model=DraftRuleValidateResponse,
    tags=["Draft Rules"],
)
async def validate_draft_rule(
    rule_id: str, request: DraftRuleValidateRequest
) -> DraftRuleValidateResponse:
    """Validate a draft rule."""
    store = get_draft_store()
    rule = store.get(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Draft rule not found: {rule_id}")

    schema_errors = []
    if rule.op not in [">", ">=", "<", "<=", "==", "in", "not_in"]:
        schema_errors.append(f"Invalid operator: {rule.op}")

    rules_to_validate = [rule]
    if request.include_existing_rules:
        rules_to_validate.extend([r for r in store.list_rules(include_archived=False) if r.id != rule_id])
        manager = get_model_manager()
        if manager.ruleset:
            rules_to_validate.extend([r for r in manager.ruleset.rules if r.status == "active"])

    test_ruleset = RuleSet(version="validation", rules=rules_to_validate)
    from rules_management.validation import validate_ruleset
    conflicts, redundancies = validate_ruleset(test_ruleset, strict=False)

    rule_conflicts = [c for c in conflicts if c.rule1_id == rule_id or c.rule2_id == rule_id]
    rule_redundancies = [r for r in redundancies if r.rule_id == rule_id]

    return DraftRuleValidateResponse(
        schema_errors=schema_errors,
        conflicts=[ConflictResponse(rule1_id=c.rule1_id, rule2_id=c.rule2_id, conflict_type=c.conflict_type, description=c.description) for c in rule_conflicts],
        redundancies=[RedundancyResponse(rule_id=r.rule_id, redundant_with=r.redundant_with, redundancy_type=r.redundancy_type, description=r.description) for r in rule_redundancies],
        is_valid=len(schema_errors) == 0 and len(rule_conflicts) == 0,
    )


@router.post(
    "/rules/draft/{rule_id}/submit",
    response_model=DraftRuleSubmitResponse,
    tags=["Draft Rules"],
)
async def submit_draft_rule(
    rule_id: str, request: DraftRuleSubmitRequest
) -> DraftRuleSubmitResponse:
    """Submit a draft rule for review."""
    store = get_draft_store()
    version_store = get_version_store()
    state_machine = RuleStateMachine(require_approval=False)

    rule = store.get(rule_id)
    if rule is None or rule.status != RuleStatus.DRAFT.value:
        raise HTTPException(status_code=404, detail="Rule not found or not in draft status")

    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.PENDING_REVIEW.value,
            actor=request.actor,
            reason=request.justification,
        )
        store.save(updated_rule)
        version_store.save(rule=updated_rule, created_by=request.actor, reason=f"Submitted: {request.justification}")
        
        return DraftRuleSubmitResponse(
            rule=DraftRuleResponse(
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
            ),
            submitted_at=datetime.now(timezone.utc).isoformat(),
            audit_id=None,
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get(
    "/rules/draft/{rule_id}/signals",
    response_model=ApprovalSignalsResponse,
    tags=["Draft Rules"],
)
async def get_approval_signals(rule_id: str) -> ApprovalSignalsResponse:
    """Get approval quality signals for a rule."""
    store = get_draft_store()
    rule = store.get(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")

    manager = get_model_manager()
    draft_rules = store.list_rules(include_archived=False)
    draft_ruleset = RuleSet(version="draft", rules=[r for r in draft_rules if r.id != rule_id])

    from api.signals import compute_approval_signals
    return compute_approval_signals(rule_id=rule_id, production_ruleset=manager.ruleset, draft_ruleset=draft_ruleset)


@router.post(
    "/rules/draft/{rule_id}/approve",
    response_model=ApproveRuleResponse,
    tags=["Draft Rules"],
)
async def approve_draft_rule(
    rule_id: str, request: ApproveRuleRequest
) -> ApproveRuleResponse:
    """Approve a pending rule."""
    store = get_draft_store()
    state_machine = create_state_machine()

    rule = store.get(rule_id)
    if rule is None or rule.status != RuleStatus.PENDING_REVIEW.value:
        raise HTTPException(status_code=404, detail="Rule not found or not pending review")

    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.APPROVED.value,
            actor=request.approver,
            reason=request.reason or "Approved",
            approver=request.approver,
        )
        store.save(updated_rule)
        get_version_store().save(rule=updated_rule, created_by=request.approver, reason=request.reason or "Approved")
        
        return ApproveRuleResponse(
            rule=DraftRuleResponse(
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
            ),
            approved_at=datetime.now(timezone.utc).isoformat(),
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/rules/{rule_id}/publish",
    response_model=PublishRuleResponse,
    tags=["Draft Rules"],
)
async def publish_rule(
    rule_id: str, request: PublishRuleRequest
) -> PublishRuleResponse:
    """Publish an approved rule to production."""
    store = get_draft_store()
    manager = get_model_manager()
    state_machine = create_state_machine()

    rule = store.get(rule_id)
    if rule is None or rule.status != RuleStatus.APPROVED.value:
        raise HTTPException(status_code=404, detail="Rule not found or not approved")

    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.ACTIVE.value,
            actor=request.actor,
            reason=request.reason or "Published",
        )
        store.save(updated_rule)
        version = get_version_store().save(rule=updated_rule, created_by=request.actor, reason=request.reason or "Published")
        
        all_active = store.list_rules(status=RuleStatus.ACTIVE.value)
        new_ruleset = RuleSet(version=f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}", rules=all_active)
        manager.update_production_ruleset(new_ruleset)

        get_audit_logger().log(
            rule_id=rule_id,
            action="RULE_PUBLISHED",
            actor=request.actor,
            before_state={"status": RuleStatus.APPROVED.value},
            after_state={"status": RuleStatus.ACTIVE.value, "version_id": version.version_id},
            reason=request.reason or "Published",
        )

        return PublishRuleResponse(
            rule=DraftRuleResponse(
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
            ),
            published_at=datetime.now(timezone.utc).isoformat(),
            version_id=version.version_id,
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/rules/draft/{rule_id}/reject",
    response_model=RejectRuleResponse,
    tags=["Draft Rules"],
)
async def reject_draft_rule(
    rule_id: str, request: RejectRuleRequest
) -> RejectRuleResponse:
    """Reject a pending rule."""
    store = get_draft_store()
    state_machine = create_state_machine()

    rule = store.get(rule_id)
    if rule is None or rule.status != RuleStatus.PENDING_REVIEW.value:
        raise HTTPException(status_code=404, detail="Rule not found or not pending review")

    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.DRAFT.value,
            actor=request.actor,
            reason=request.reason,
        )
        store.save(updated_rule)
        get_version_store().save(rule=updated_rule, created_by=request.actor, reason=f"Rejected: {request.reason}")
        
        return RejectRuleResponse(
            rule=DraftRuleResponse(
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
            ),
            rejected_at=datetime.now(timezone.utc).isoformat(),
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/rules/{rule_id}/activate",
    response_model=ActivateRuleResponse,
    tags=["Rules"],
)
async def activate_rule(
    rule_id: str, request: ActivateRuleRequest
) -> ActivateRuleResponse:
    """Activate a rule."""
    store = get_draft_store()
    state_machine = create_state_machine()
    manager = get_model_manager()

    rule = store.get(rule_id)
    if rule is None and manager.ruleset:
        rule = next((r for r in manager.ruleset.rules if r.id == rule_id), None)

    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")

    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.ACTIVE.value,
            actor=request.actor,
            reason=request.reason,
            approver=request.approver,
        )
        store.save(updated_rule)
        get_version_store().save(rule=updated_rule, created_by=request.actor, reason=request.reason)
        
        return ActivateRuleResponse(
            rule=DraftRuleResponse(
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
            ),
            activated_at=datetime.now(timezone.utc).isoformat(),
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/rules/{rule_id}/disable",
    response_model=DisableRuleResponse,
    tags=["Rules"],
)
async def disable_rule(
    rule_id: str, request: DisableRuleRequest
) -> DisableRuleResponse:
    """Disable a rule."""
    store = get_draft_store()
    state_machine = create_state_machine()
    manager = get_model_manager()

    rule = store.get(rule_id)
    if rule is None and manager.ruleset:
        rule = next((r for r in manager.ruleset.rules if r.id == rule_id), None)

    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")

    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.DISABLED.value,
            actor=request.actor,
            reason=request.reason or "Disabled",
        )
        store.save(updated_rule)
        get_version_store().save(rule=updated_rule, created_by=request.actor, reason=request.reason or "Disabled")
        
        return DisableRuleResponse(
            rule=DraftRuleResponse(
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
            ),
            disabled_at=datetime.now(timezone.utc).isoformat(),
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/rules/{rule_id}/shadow",
    response_model=ShadowRuleResponse,
    tags=["Rules"],
)
async def shadow_rule(rule_id: str, request: ShadowRuleRequest) -> ShadowRuleResponse:
    """Move a rule to shadow mode."""
    store = get_draft_store()
    state_machine = create_state_machine()
    manager = get_model_manager()

    rule = store.get(rule_id)
    if rule is None and manager.ruleset:
        rule = next((r for r in manager.ruleset.rules if r.id == rule_id), None)

    if rule is None:
        raise HTTPException(status_code=404, detail="Rule not found")

    try:
        updated_rule = state_machine.transition(
            rule=rule,
            new_status=RuleStatus.SHADOW.value,
            actor=request.actor,
            reason=request.reason or "Shadowed",
        )
        store.save(updated_rule)
        get_version_store().save(rule=updated_rule, created_by=request.actor, reason=request.reason or "Shadowed")
        
        return ShadowRuleResponse(
            rule=DraftRuleResponse(
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
            ),
            shadowed_at=datetime.now(timezone.utc).isoformat(),
        )
    except TransitionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/rules/{rule_id}/versions",
    response_model=RuleVersionListResponse,
    tags=["Rules"],
)
async def list_rule_versions(rule_id: str) -> RuleVersionListResponse:
    """List all versions of a rule."""
    versions = get_version_store().list_versions(rule_id)
    return RuleVersionListResponse(
        versions=[
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
        ],
        total=len(versions)
    )


@router.get(
    "/rules/{rule_id}/versions/{version_id}",
    response_model=RuleVersionResponse,
    tags=["Rules"],
)
async def get_rule_version(rule_id: str, version_id: str) -> RuleVersionResponse:
    """Get a specific version of a rule."""
    version = get_version_store().get_version(rule_id, version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Version not found")

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


@router.get(
    "/rules/{rule_id}/diff",
    response_model=RuleDiffResponse,
    tags=["Rules"],
)
async def get_rule_diff(
    rule_id: str,
    version_a: str | None = Query(None),
    version_b: str | None = Query(None),
) -> RuleDiffResponse:
    """Compare two versions of a rule."""
    version_store = get_version_store()
    versions = version_store.list_versions(rule_id)
    if not versions:
        raise HTTPException(status_code=404, detail="Rule not found")

    ver_a = version_store.get_version(rule_id, version_a) if version_a else versions[-1]
    if not ver_a:
        raise HTTPException(status_code=404, detail="Version A not found")

    if version_b:
        ver_b = version_store.get_version(rule_id, version_b)
    else:
        idx = next((i for i, v in enumerate(versions) if v.version_id == ver_a.version_id), 0)
        ver_b = versions[idx - 1] if idx > 0 else None

    if not ver_b:
        raise HTTPException(status_code=400, detail="Version B not found or no predecessor")

    diff_result = diff_rule_versions(ver_a, ver_b)
    return RuleDiffResponse(
        version_a_id=diff_result.version_a_id,
        version_b_id=diff_result.version_b_id,
        rule_id=diff_result.rule_id,
        changes=[RuleFieldChangeResponse(field_name=c.field_name, change_type=c.change_type, old_value=c.old_value, new_value=c.new_value) for c in diff_result.changes],
        is_breaking=diff_result.is_breaking,
        version_a_timestamp=diff_result.version_a_timestamp.isoformat() if diff_result.version_a_timestamp else "",
        version_b_timestamp=diff_result.version_b_timestamp.isoformat() if diff_result.version_b_timestamp else "",
        version_a_created_by=diff_result.version_a_created_by,
        version_b_created_by=diff_result.version_b_created_by,
    )


@router.post(
    "/rules/{rule_id}/versions/{version_id}/rollback",
    response_model=RollbackRuleResponse,
    tags=["Rules"],
)
async def rollback_rule_version(
    rule_id: str, version_id: str, request: RollbackRuleRequest
) -> RollbackRuleResponse:
    """Rollback a rule to a previous version."""
    version_store = get_version_store()
    try:
        new_version = version_store.rollback(rule_id=rule_id, version_id=version_id, rolled_back_by=request.actor, reason=request.reason)
        get_draft_store().save(new_version.rule)
        
        return RollbackRuleResponse(
            rule=DraftRuleResponse(
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
            ),
            version_id=new_version.version_id,
            rolled_back_to=version_id,
            rolled_back_at=datetime.now(timezone.utc).isoformat(),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/audit/logs",
    response_model=AuditLogQueryResponse,
    tags=["Audit"],
)
async def query_audit_logs(
    rule_id: str | None = Query(None),
    actor: str | None = Query(None),
    action: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
) -> AuditLogQueryResponse:
    """Query audit logs."""
    audit_logger = get_audit_logger()
    start_dt = datetime.fromisoformat(start_date) if start_date else None
    end_dt = datetime.fromisoformat(end_date) if end_date else None
    
    records = audit_logger.query(rule_id=rule_id, actor=actor, action=action, start_date=start_dt, end_date=end_dt)
    return AuditLogQueryResponse(
        records=[AuditRecordResponse(rule_id=r.rule_id, action=r.action, actor=r.actor, timestamp=r.timestamp.isoformat(), before_state=r.before_state, after_state=r.after_state, reason=r.reason) for r in records],
        total=len(records)
    )


@router.get(
    "/audit/rules/{rule_id}/history",
    response_model=AuditLogQueryResponse,
    tags=["Audit"],
)
async def get_rule_audit_history(rule_id: str) -> AuditLogQueryResponse:
    """Get audit history for a rule."""
    records = get_audit_logger().get_rule_history(rule_id)
    return AuditLogQueryResponse(
        records=[AuditRecordResponse(rule_id=r.rule_id, action=r.action, actor=r.actor, timestamp=r.timestamp.isoformat(), before_state=r.before_state, after_state=r.after_state, reason=r.reason) for r in records],
        total=len(records)
    )


@router.get(
    "/analytics/rules/{rule_id}",
    response_model=RuleAnalyticsResponse,
    tags=["Analytics"],
)
async def get_rule_analytics(
    rule_id: str,
    days: int = Query(7, ge=1, le=90),
) -> RuleAnalyticsResponse:
    """Get rule analytics."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)
    metrics = get_metrics_collector().get_rule_metrics(rule_id, start_date, end_date)
    
    manager = get_model_manager()
    rule = next((r for r in manager.ruleset.rules if r.id == rule_id), None) if manager.ruleset else None
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found in active ruleset")

    health_report = RuleHealthEvaluator().evaluate(rule, metrics, 1000)
    return RuleAnalyticsResponse(
        rule_id=rule_id,
        health=RuleHealthResponse(rule_id=rule_id, status=health_report.status.value, reason=health_report.reason, metrics=metrics.to_dict()),
        statistics={"mean_score_delta": metrics.mean_score_delta, "mean_latency_ms": metrics.mean_execution_time_ms, "total_matches": metrics.production_matches + metrics.shadow_matches},
        history_summary=[]
    )


@router.get(
    "/rules/{rule_id}/readiness",
    response_model=ReadinessReportResponse,
    tags=["Rules"],
)
async def check_rule_readiness(rule_id: str) -> ReadinessReportResponse:
    """Check rule readiness."""
    manager = get_model_manager()
    rule = next((r for r in manager.ruleset.rules if r.id == rule_id), None) if manager.ruleset else get_draft_store().get(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")

    metrics = get_metrics_collector().get_rule_metrics(rule_id, datetime.now(timezone.utc) - timedelta(days=7), datetime.now(timezone.utc))
    report = ReadinessEvaluator(audit_logger=get_audit_logger()).evaluate(rule, metrics, 1000)
    
    return ReadinessReportResponse(
        rule_id=report.rule_id,
        timestamp=report.timestamp.isoformat(),
        overall_status=report.overall_status.value,
        checks=[{"policy_type": c.policy_type.value, "name": c.name, "status": c.status.value, "message": c.message, "details": c.details} for c in report.checks],
    )


@router.get(
    "/analytics/attribution",
    response_model=RuleAttributionResponse,
    tags=["Analytics"],
)
async def get_rule_attribution(
    rule_id: str,
    days: int = Query(7, ge=1, le=90),
) -> RuleAttributionResponse:
    """Get attribution metrics."""
    service = AttributionService()
    attribution = service.get_rule_attribution(rule_id, datetime.now(timezone.utc) - timedelta(days=days), datetime.now(timezone.utc))
    if not attribution:
        raise HTTPException(status_code=404, detail="No attribution data found")

    return RuleAttributionResponse(
        rule_id=attribution.rule_id,
        total_matches=attribution.total_matches,
        mean_model_score=attribution.mean_model_score,
        mean_final_score=attribution.mean_final_score,
        mean_impact=attribution.mean_impact,
        net_impact=attribution.net_impact,
    )


@router.get(
    "/analytics/relationships",
    response_model=DatasetRelationshipsResponse,
    tags=["Analytics"],
)
async def get_dataset_relationships(
    sample_size: int = Query(default=500, ge=10, le=2000),
    target_column: str = Query(default="is_fraudulent"),
) -> DatasetRelationshipsResponse:
    """Compute feature relationships."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    client = get_crud_client()
    try:
        resp = client.get_feature_sample(sample_size=sample_size, stratify=True)
        samples = MessageToDict(resp, preserving_proto_field_name=True, always_print_fields_with_no_presence=True, use_integers_for_enums=True).get("samples", [])
        if not samples:
            return DatasetRelationshipsResponse(relationships=[], target_column=target_column)

        df = pd.DataFrame(samples)
        features = [c for c in df.columns if c not in ["record_id", "user_id", "snapshot_id", target_column]]
        relationships = []

        if target_column in df.columns:
            for col in features:
                if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_column]):
                    corr, _ = stats.pearsonr(df[col].fillna(0), df[target_column].fillna(0))
                    relationships.append(RelationshipMetric(feature_a=col, feature_b=target_column, metric_type="pearson", value=float(corr) if not np.isnan(corr) else 0.0))
        
        relationships.sort(key=lambda x: abs(x.value), reverse=True)
        return DatasetRelationshipsResponse(relationships=relationships[:20], target_column=target_column)
    except Exception as e:
        logger.error(f"Failed relationships: {e}")
        raise analytics_http_exception(e)


@router.get(
    "/analytics/correlations",
    response_model=DatasetCorrelationsResponse,
    tags=["Analytics"],
)
async def get_dataset_correlations(
    sample_size: int = Query(default=1000, ge=100, le=5000),
) -> DatasetCorrelationsResponse:
    """Compute dataset correlations."""
    import numpy as np
    import pandas as pd
    from scipy import stats

    client = get_crud_client()
    try:
        resp = client.get_feature_sample(sample_size=sample_size, stratify=True)
        samples = MessageToDict(resp, preserving_proto_field_name=True, always_print_fields_with_no_presence=True, use_integers_for_enums=True).get("samples", [])
        if not samples:
            return DatasetCorrelationsResponse(pearson=[], spearman=[], cramers_v=[], numeric_columns=[], categorical_columns=[])

        df = pd.DataFrame(samples)
        numeric_cols = [c for c in df.columns if c not in ["record_id", "user_id", "snapshot_id"] and pd.api.types.is_numeric_dtype(df[c])]
        
        pearson_pairs = []
        if len(numeric_cols) >= 2:
            p_corr = df[numeric_cols].corr(method="pearson")
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    val = p_corr.iloc[i, j]
                    if not np.isnan(val):
                        pearson_pairs.append(CorrelationPair(feature_a=numeric_cols[i], feature_b=numeric_cols[j], value=float(val)))

        return DatasetCorrelationsResponse(pearson=pearson_pairs, spearman=[], cramers_v=[], numeric_columns=numeric_cols, categorical_columns=[])
    except Exception as e:
        logger.error(f"Failed correlations: {e}")
        raise analytics_http_exception(e)