import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from forecast_server.drift_cache import get_drift_cache
from forecast_server.model_manager import get_model_manager
from forecast_server.services import get_forecaster
from training_server.audit import get_audit_logger
from training_server.schemas import (
    DeployModelRequest,
    DeployModelResponse,
    DriftStatusResponse,
    FeatureDriftDetail,
    PredictResponse,
    ScoreDistributionItem,
    ScoreDistributionResponse,
    SignalRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=dict, tags=["System"])
async def health_check() -> dict:
    """Check API health status."""
    manager = get_model_manager()
    forecaster = get_forecaster()

    # Use model manager version if available, otherwise fall back to forecaster
    version = (
        manager.model_version if manager.model_loaded else forecaster.model_version
    )

    return {
        "status": "healthy",
        "model_loaded": manager.model_loaded,
        "version": version,
    }


@router.get(
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
    """Check feature distribution drift between reference and live data."""
    import time

    from training_server.detect_drift import (
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
    """Build DriftStatusResponse from detect_drift result."""
    from training_server.detect_drift import (
        PSI_THRESHOLD_CRITICAL,
        PSI_THRESHOLD_WARNING,
    )

    if overall_status is None:
        if "error" in result:
            overall_status = "unknown"
        elif result.get("drift_detected", False):
            overall_status = "fail"
        else:
            has_warning = any(
                details.get("status") == "WARNING"
                for details in result.get("features", {}).values()
            )
            if has_warning:
                overall_status = "warn"
            else:
                overall_status = "ok"

    top_features = []
    for feature_name, details in result.get("features", {}).items():
        top_features.append(
            FeatureDriftDetail(
                feature=feature_name,
                psi=details.get("psi", 0.0),
                status=details.get("status", "OK"),
            )
        )
    top_features.sort(key=lambda x: x.psi, reverse=True)

    reference_window = "Production model reference data"
    if result.get("reference_size", 0) > 0:
        reference_window = (
            f"Production model reference data ({result['reference_size']} samples)"
        )

    hours_analyzed = result.get("hours_analyzed", 24)
    current_window = f"Last {hours_analyzed} hours"
    if result.get("live_size", 0) > 0:
        current_window = f"Last {hours_analyzed} hours ({result['live_size']} samples)"

    # C1: Structured logging for alerts
    alerts = result.get("alerts", [])
    for alert in alerts:
        log_level = (
            logging.ERROR if alert["severity"] == "critical" else logging.WARNING
        )
        logger.log(
            log_level,
            "DRIFT ALERT: "
            f"{alert['severity'].upper()} drift detected for feature "
            f"'{alert['feature']}' "
            f"(PSI={alert['psi']}, threshold={alert['threshold']})",
        )

    return DriftStatusResponse(
        status=overall_status,
        computed_at=result.get("timestamp", datetime.now(timezone.utc).isoformat()),
        cached=cached,
        reference_window=reference_window,
        current_window=current_window,
        reference_size=result.get("reference_size", 0),
        live_size=result.get("live_size", 0),
        top_features=top_features,
        alerts=alerts,
        thresholds={
            "warn": PSI_THRESHOLD_WARNING,
            "fail": PSI_THRESHOLD_CRITICAL,
        },
        error=result.get("error"),
    )


@router.get(
    "/monitoring/score-distribution",
    response_model=ScoreDistributionResponse,
    tags=["Monitoring"],
    summary="Check final score distribution drift",
)
async def get_score_distribution(
    hours: int = Query(
        default=24, ge=1, le=168, description="Hours of live data to analyze"
    ),
) -> ScoreDistributionResponse:
    """Check score distribution drift between training baseline and live data."""
    import numpy as np

    from training_server.crud_client import get_crud_client

    manager = get_model_manager()
    baseline = manager.baseline_distribution

    # Fetch live scores
    client = get_crud_client()
    try:
        resp = client.get_inference_scores(hours=hours)
        live_scores = np.array(resp.scores)
    except Exception as e:
        logger.error(f"Failed to fetch inference scores: {e}")
        live_scores = np.array([])

    # Define buckets (C3): [1-10, 11-30, 31-70, 71-90, 91-99]
    buckets = [1, 11, 31, 71, 91, 100]

    if len(live_scores) > 0:
        counts, _ = np.histogram(live_scores, bins=buckets)
        observed_ratios = counts / len(live_scores)
    else:
        counts = np.zeros(len(buckets) - 1)
        observed_ratios = np.zeros(len(buckets) - 1)

    baseline_ratios = (
        np.array(baseline["ratios"])
        if baseline
        else np.ones(len(buckets) - 1) / (len(buckets) - 1)
    )

    # Compute Jensen-Shannon Divergence
    def kl_div(p, q):
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        return np.sum(p * np.log(p / q))

    m = 0.5 * (baseline_ratios + observed_ratios)
    js_div = 0.5 * (kl_div(baseline_ratios, m) + kl_div(observed_ratios, m))

    # Detect shift: any bucket > 2x baseline
    shift_detected = False
    distribution_items = []
    for i in range(len(buckets) - 1):
        b_ratio = float(baseline_ratios[i])
        o_ratio = float(observed_ratios[i])
        if b_ratio > 0 and o_ratio > 2 * b_ratio:
            shift_detected = True

        distribution_items.append(
            ScoreDistributionItem(
                bucket=[buckets[i], buckets[i + 1] - 1],
                baseline_ratio=b_ratio,
                observed_ratio=o_ratio,
                observed_count=int(counts[i]),
            )
        )

    return ScoreDistributionResponse(
        computed_at=datetime.now(timezone.utc).isoformat(),
        observed_size=len(live_scores),
        baseline_size=baseline["total"] if baseline else None,
        divergence=float(js_div),
        divergence_metric="JS",
        distribution=distribution_items,
        shift_detected=shift_detected,
    )


@router.post("/reload-model", tags=["System"])
async def reload_model() -> dict:
    """Reload the production model from MLflow."""
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


@router.post(
    "/models/deploy",
    response_model=DeployModelResponse,
    tags=["System"],
    summary="Deploy a model to production",
)
async def deploy_model(request: DeployModelRequest) -> DeployModelResponse:
    """Deploy a model to production."""
    import os

    import mlflow

    manager = get_model_manager()
    audit_logger = get_audit_logger()

    previous_version = manager.model_version if manager.model_loaded else None
    success = manager.load_production_model()

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to load production model from MLflow",
        )

    deployed_version = manager.model_version
    deployed_at = datetime.now(timezone.utc).isoformat()

    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5005")
        )
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(
            "name='ach-fraud-detection' AND current_stage='Production'"
        )
        for v in versions:
            if v.version == deployed_version.lstrip("v"):
                client.set_model_version_tag(
                    "ach-fraud-detection",
                    v.version,
                    "deployed_at",
                    deployed_at,
                )
                client.set_model_version_tag(
                    "ach-fraud-detection",
                    v.version,
                    "deployed_by",
                    request.actor,
                )
                if request.reason:
                    client.set_model_version_tag(
                        "ach-fraud-detection",
                        v.version,
                        "deployment_reason",
                        request.reason,
                    )
                break
    except Exception as e:
        logger.warning(f"Failed to set MLflow deployment tags: {e}")

    audit_logger.log(
        rule_id=f"model:{deployed_version}",
        action="MODEL_DEPLOYED",
        actor=request.actor,
        before_state={"model_version": previous_version} if previous_version else None,
        after_state={
            "model_version": deployed_version,
            "deployed_at": deployed_at,
        },
        reason=request.reason or "Model deployed to production",
    )

    return DeployModelResponse(
        success=True,
        model_version=deployed_version,
        deployed_at=deployed_at,
        previous_version=previous_version,
    )


@router.post(
    "/predict/signal",
    response_model=PredictResponse,
    tags=["Forecasting"],
    summary="Get model prediction only",
)
async def predict_signal(request: SignalRequest) -> PredictResponse:
    """Predict fraud risk only (no rules)."""
    try:
        forecaster = get_forecaster()
        result = forecaster.predict(request)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e!s}",
        ) from e
