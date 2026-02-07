import logging

import grpc
from google.protobuf.struct_pb2 import Struct

from forecast.drift_cache import get_drift_cache
from forecast.model_manager import get_model_manager
from forecast.services import get_forecaster
from forecast.v1 import forecast_pb2, forecast_pb2_grpc

logger = logging.getLogger(__name__)


class ForecastService(forecast_pb2_grpc.ForecastServiceServicer):
    def GetHealth(self, request, context):  # noqa: N802
        manager = get_model_manager()
        forecaster = get_forecaster()
        version = (
            manager.model_version if manager.model_loaded else forecaster.model_version
        )

        return forecast_pb2.GetHealthResponse(
            status="healthy",
            components={
                "model_loaded": "true" if manager.model_loaded else "false",
                "version": version or "unknown",
            },
        )

    def GetDriftMonitoring(self, request, context):  # noqa: N802
        from training.detect_drift import PSI_THRESHOLD_CRITICAL, detect_drift

        hours = request.hours or 24
        threshold = request.threshold or PSI_THRESHOLD_CRITICAL

        cache = get_drift_cache()
        result = None
        if not request.force_refresh:
            result = cache.get(hours, threshold)

        if result is None:
            try:
                result = detect_drift(hours=hours, threshold=threshold)
                cache.set(hours, threshold, result)
            except Exception as e:
                logger.exception("Drift detection failed")
                context.abort(grpc.StatusCode.INTERNAL, f"Drift detection failed: {e}")

        # Map result to proto
        metrics = Struct()
        if "features" in result:
            metrics.update(result["features"])

        return forecast_pb2.GetDriftMonitoringResponse(
            drift_detected=result.get("drift_detected", False),
            drift_score=result.get("overall_psi", 0.0),
            metrics=metrics,
        )

    def GetScoreDistribution(self, request, context):  # noqa: N802
        import numpy as np

        from training.crud_client import get_crud_client

        hours = request.hours or 24

        client = get_crud_client()
        try:
            resp = client.get_inference_scores(hours=hours)
            live_scores = np.array(resp.scores)
        except Exception as e:
            logger.error(f"Failed to fetch inference scores: {e}")
            live_scores = np.array([])

        buckets = [1, 11, 31, 71, 91, 100]
        if len(live_scores) > 0:
            counts, _ = np.histogram(live_scores, bins=buckets)
            observed_ratios = counts / len(live_scores)
        else:
            counts = np.zeros(len(buckets) - 1)
            observed_ratios = np.zeros(len(buckets) - 1)

        manager = get_model_manager()
        baseline = manager.baseline_distribution
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
        for i in range(len(buckets) - 1):
            if baseline_ratios[i] > 0 and observed_ratios[i] > 2 * baseline_ratios[i]:
                shift_detected = True
                break

        return forecast_pb2.GetScoreDistributionResponse(
            scores=[int(s) for s in live_scores],
            distribution={
                f"{buckets[i]}-{buckets[i + 1] - 1}": int(counts[i])
                for i in range(len(buckets) - 1)
            },
            divergence=float(js_div),
            divergence_metric="JS",
            shift_detected=shift_detected,
        )

    def ReloadModel(self, request, context):  # noqa: N802
        manager = get_model_manager()
        success = manager.load_production_model()
        return forecast_pb2.ReloadModelResponse(
            success=success,
            model_version=manager.model_version if success else "",
        )

    def DeployModel(self, request, context):  # noqa: N802
        from datetime import datetime, timezone

        from training.audit import get_audit_logger

        manager = get_model_manager()
        audit_logger = get_audit_logger()

        success = manager.load_production_model()

        if not success:
            context.abort(
                grpc.StatusCode.INTERNAL, "Failed to load production model from MLflow"
            )

        deployed_version = manager.model_version
        deployed_at = datetime.now(timezone.utc).isoformat()

        audit_logger.log(
            rule_id=f"model:{deployed_version}",
            action="MODEL_DEPLOYED",
            actor=request.actor or "gRPC-system",
            after_state={"model_version": deployed_version, "deployed_at": deployed_at},
            reason=request.reason or "Model deployed via gRPC",
        )

        return forecast_pb2.DeployModelResponse(
            success=True,
            message=f"Model {deployed_version} deployed successfully",
            model_version=deployed_version,
            deployed_at=deployed_at,
        )

    def PredictSignal(self, request, context):  # noqa: N802
        forecaster = get_forecaster()
        from training.schemas import SignalRequest

        # Map proto to internal schema
        internal_req = SignalRequest(
            user_id=request.user_id,
            amount=request.amount,
            currency=request.currency,
            client_transaction_id=request.client_transaction_id,
        )

        try:
            result = forecaster.predict(internal_req)
            return forecast_pb2.PredictSignalResponse(
                probability=result["probability"],
                model_version=result["model_version"],
                run_id=result.get("run_id", ""),
            )
        except Exception as e:
            logger.exception("Prediction failed")
            context.abort(grpc.StatusCode.INTERNAL, f"Prediction failed: {e}")
