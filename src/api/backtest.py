"""Backtesting infrastructure for rule evaluation via Analytics service."""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp

from api.crud_client import get_crud_client
from api.rules import RuleSet
from api.schemas import BacktestDelta

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Metrics computed from a backtest run."""

    total_records: int
    matched_count: int
    match_rate: float
    score_distribution: dict[str, int]  # score range -> count
    score_mean: float
    score_std: float
    score_min: int
    score_max: int
    rejected_count: int
    rejected_rate: float


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    job_id: str
    rule_id: str | None
    ruleset_version: str
    start_date: datetime
    end_date: datetime
    metrics: BacktestMetrics
    completed_at: datetime
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        data["end_date"] = self.end_date.isoformat()
        data["completed_at"] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BacktestResult":
        """Create from dictionary."""
        for field in ["start_date", "end_date", "completed_at"]:
            if isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        if isinstance(data["metrics"], dict):
            data["metrics"] = BacktestMetrics(**data["metrics"])
        return cls(**data)


class BacktestRunner:
    """Runs backtests by replaying historical features from Analytics service."""

    def __init__(self, db_session=None):
        """Initialize backtest runner.

        Args:
            db_session: Ignored in compute-only mode.
        """
        pass

    def run_backtest(
        self,
        ruleset: RuleSet,
        start_date: datetime,
        end_date: datetime,
        base_score: int = 50,
        rule_id: str | None = None,
    ) -> BacktestResult:
        """Run a backtest on historical data."""
        import os
        from dataclasses import asdict

        from api.gateway_client import get_gateway_client

        job_id = f"backtest_{uuid.uuid4().hex[:12]}"
        logger.info(f"Starting backtest {job_id} for ruleset {ruleset.version}")

        client = get_gateway_client()
        ruleset_dict = asdict(ruleset)

        try:
            # Fetch historical features via Analytics service
            features_list = self._fetch_historical_features(start_date, end_date)

            if not features_list:
                return self._empty_result(
                    job_id, ruleset.version, start_date, end_date, rule_id
                )

            # Replay features through rules
            scores = []
            matched_counts = []
            rejected_counts = []

            for i, feature_dict in enumerate(features_list):
                result = client.evaluate_rules(
                    features=feature_dict,
                    base_score=base_score,
                    ruleset=ruleset_dict,
                    request_id=f"{job_id}_{i}",
                )
                scores.append(result["final_score"])
                matched_counts.append(len(result["matched_rules"]))
                rejected_counts.append(1 if result.get("rejected") else 0)

            # Compute metrics
            metrics = self._compute_metrics(
                total_records=len(features_list),
                scores=scores,
                matched_counts=matched_counts,
                rejected_counts=rejected_counts,
            )

            result = BacktestResult(
                job_id=job_id,
                rule_id=rule_id,
                ruleset_version=ruleset.version,
                start_date=start_date,
                end_date=end_date,
                metrics=metrics,
                completed_at=datetime.now(timezone.utc),
            )

            # Persist result via Analytics service
            self._save_result(result)

            logger.info(f"Backtest {job_id} completed and saved to Analytics service")
            return result

        except Exception as e:
            logger.error(f"Backtest {job_id} failed: {e}", exc_info=True)
            return self._error_result(
                job_id, ruleset.version, start_date, end_date, rule_id, str(e)
            )

    def _fetch_historical_features(
        self, start_date: datetime, end_date: datetime
    ) -> list[dict[str, Any]]:
        """Fetch historical features from Analytics service."""
        from api.proto.proto.crud.v1 import analytics_pb2

        start_ts = Timestamp()
        start_ts.FromDatetime(start_date)
        end_ts = Timestamp()
        end_ts.FromDatetime(end_date)

        client = get_crud_client()
        resp = client.stub.GetBacktestFeatures(
            analytics_pb2.GetBacktestFeaturesRequest(
                start_date=start_ts,
                end_date=end_ts,
            ),
            timeout=client.timeout_seconds,
        )

        features_list = []
        for f in resp.features:
            feature_dict = {
                "velocity_24h": f.velocity_24h,
                "amount_to_avg_ratio_30d": f.amount_to_avg_ratio_30d,
                "balance_volatility_z_score": f.balance_volatility_z_score,
            }
            if f.experimental_signals_json:
                try:
                    exp = json.loads(f.experimental_signals_json)
                    if isinstance(exp, dict):
                        feature_dict.update(exp)
                except json.JSONDecodeError:
                    pass
            features_list.append(feature_dict)
        return features_list

    def _save_result(self, result: BacktestResult) -> None:
        """Save backtest result via Analytics service."""
        from api.proto.proto.crud.v1 import analytics_pb2

        client = get_crud_client()

        start_ts = Timestamp()
        start_ts.FromDatetime(result.start_date)
        end_ts = Timestamp()
        end_ts.FromDatetime(result.end_date)
        comp_ts = Timestamp()
        completed_at = result.completed_at
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)
        comp_ts.FromDatetime(completed_at)

        m = result.metrics
        metrics_pb = analytics_pb2.BacktestMetrics(
            total_records=m.total_records,
            matched_count=m.matched_count,
            match_rate=m.match_rate,
            score_distribution=m.score_distribution,
            score_mean=m.score_mean,
            score_std=m.score_std,
            score_min=m.score_min,
            score_max=m.score_max,
            rejected_count=m.rejected_count,
            rejected_rate=m.rejected_rate,
        )

        res_pb = analytics_pb2.BacktestResult(
            job_id=result.job_id,
            rule_id=result.rule_id or "",
            ruleset_version=result.ruleset_version,
            start_date=start_ts,
            end_date=end_ts,
            metrics=metrics_pb,
            completed_at=comp_ts,
            error=result.error or "",
        )

        client.stub.SaveBacktestResult(
            analytics_pb2.SaveBacktestResultRequest(result=res_pb),
            timeout=client.timeout_seconds,
        )

    def _empty_result(self, job_id, version, start, end, rule_id):
        return BacktestResult(
            job_id=job_id,
            rule_id=rule_id,
            ruleset_version=version,
            start_date=start,
            end_date=end,
            metrics=BacktestMetrics(0, 0, 0.0, {}, 0.0, 0.0, 0, 0, 0, 0.0),
            completed_at=datetime.now(timezone.utc),
        )

    def _error_result(self, job_id, version, start, end, rule_id, error_msg):
        res = self._empty_result(job_id, version, start, end, rule_id)
        res.error = error_msg
        return res

    def _compute_metrics(
        self, total_records, scores, matched_counts, rejected_counts
    ) -> BacktestMetrics:
        if total_records == 0:
            return BacktestMetrics(0, 0, 0.0, {}, 0.0, 0.0, 0, 0, 0, 0.0)

        scores_array = np.array(scores)
        matched_count = sum(1 for c in matched_counts if c > 0)
        rejected_count = sum(rejected_counts)

        score_ranges = {"1-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-99": 0}
        for score in scores:
            if score <= 20:
                score_ranges["1-20"] += 1
            elif score <= 40:
                score_ranges["21-40"] += 1
            elif score <= 60:
                score_ranges["41-60"] += 1
            elif score <= 80:
                score_ranges["61-80"] += 1
            else:
                score_ranges["81-99"] += 1

        return BacktestMetrics(
            total_records=total_records,
            matched_count=matched_count,
            match_rate=matched_count / total_records,
            score_distribution=score_ranges,
            score_mean=float(np.mean(scores_array)),
            score_std=float(np.std(scores_array)),
            score_min=int(np.min(scores_array)),
            score_max=int(np.max(scores_array)),
            rejected_count=rejected_count,
            rejected_rate=rejected_count / total_records,
        )


class BacktestStore:
    """Retrieves backtest results from Analytics service."""

    def __init__(self, storage_path=None):
        """Initialize. storage_path is ignored."""
        pass

    def save(self, result: BacktestResult) -> None:
        """Save is handled by BacktestRunner directly via Analytics service."""
        pass

    def get(self, job_id: str) -> BacktestResult | None:
        """Get result from Analytics service."""
        from api.proto.proto.crud.v1 import analytics_pb2

        client = get_crud_client()
        try:
            resp = client.stub.GetBacktestResult(
                analytics_pb2.GetBacktestResultRequest(job_id=job_id),
                timeout=client.timeout_seconds,
            )
            return self._from_pb(resp.result)
        except Exception:
            return None

    def list_results(
        self, rule_id=None, start_date=None, end_date=None
    ) -> list[BacktestResult]:
        """List results from Analytics service."""
        from api.proto.proto.crud.v1 import analytics_pb2

        client = get_crud_client()

        start_ts = None
        if start_date:
            start_ts = Timestamp()
            start_ts.FromDatetime(
                start_date
                if start_date.tzinfo
                else start_date.replace(tzinfo=timezone.utc)
            )

        end_ts = None
        if end_date:
            end_ts = Timestamp()
            end_ts.FromDatetime(
                end_date if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)
            )

        req = analytics_pb2.ListBacktestResultsRequest(
            rule_id=rule_id or "", start_date=start_ts, end_date=end_ts
        )
        resp = client.stub.ListBacktestResults(
            req,
            timeout=client.timeout_seconds,
        )
        return [self._from_pb(r) for r in resp.results]

    def _from_pb(self, r) -> BacktestResult:
        """Convert proto BacktestResult to local dataclass."""
        m = r.metrics
        metrics = BacktestMetrics(
            total_records=m.total_records,
            matched_count=m.matched_count,
            match_rate=m.match_rate,
            score_distribution=dict(m.score_distribution),
            score_mean=m.score_mean,
            score_std=m.score_std,
            score_min=m.score_min,
            score_max=m.score_max,
            rejected_count=m.rejected_count,
            rejected_rate=m.rejected_rate,
        )
        return BacktestResult(
            job_id=r.job_id,
            rule_id=r.rule_id if r.rule_id else None,
            ruleset_version=r.ruleset_version,
            start_date=r.start_date.ToDatetime(),
            end_date=r.end_date.ToDatetime(),
            metrics=metrics,
            completed_at=r.completed_at.ToDatetime(),
            error=r.error if r.error else None,
        )


class BacktestComparator:
    """Compare backtest results and compute deltas."""

    def compute_delta(
        self, base: BacktestMetrics, candidate: BacktestMetrics
    ) -> BacktestDelta:
        return BacktestDelta(
            match_rate_delta=candidate.match_rate - base.match_rate,
            rejected_rate_delta=candidate.rejected_rate - base.rejected_rate,
            score_mean_delta=candidate.score_mean - base.score_mean,
            score_std_delta=candidate.score_std - base.score_std,
            matched_count_delta=candidate.matched_count - base.matched_count,
            rejected_count_delta=candidate.rejected_count - base.rejected_count,
        )


_global_backtest_store: BacktestStore | None = None


def get_backtest_store() -> BacktestStore:
    """Get the global backtest store instance."""
    global _global_backtest_store
    if _global_backtest_store is None:
        _global_backtest_store = BacktestStore()
    return _global_backtest_store
