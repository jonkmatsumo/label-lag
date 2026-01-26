"""Backtesting infrastructure for rule evaluation."""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import text

from api.rules import RuleSet, evaluate_rules
from api.schemas import BacktestDelta
from synthetic_pipeline.db.session import DatabaseSession

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
    rule_id: str | None  # If testing single rule, None if testing ruleset
    ruleset_version: str
    start_date: datetime
    end_date: datetime
    metrics: BacktestMetrics
    completed_at: datetime
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["start_date"] = self.start_date.isoformat()
        data["end_date"] = self.end_date.isoformat()
        data["completed_at"] = self.completed_at.isoformat()
        data["metrics"] = asdict(self.metrics)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BacktestResult":
        """Create from dictionary."""
        # Convert ISO strings back to datetime
        for field in ["start_date", "end_date", "completed_at"]:
            if isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])

        # Reconstruct metrics
        data["metrics"] = BacktestMetrics(**data["metrics"])

        return cls(**data)


class BacktestRunner:
    """Runs backtests by replaying historical features through rules."""

    def __init__(self, db_session: DatabaseSession | None = None):
        """Initialize backtest runner.

        Args:
            db_session: Database session. If None, creates a new one.
        """
        self.db_session = db_session or DatabaseSession()

    def run_backtest(
        self,
        ruleset: RuleSet,
        start_date: datetime,
        end_date: datetime,
        base_score: int = 50,
        rule_id: str | None = None,
    ) -> BacktestResult:
        """Run a backtest on historical data.

        Args:
            ruleset: RuleSet to test.
            start_date: Start of date range.
            end_date: End of date range.
            base_score: Base score to use before rule application.
            rule_id: Optional rule ID if testing a single rule.

        Returns:
            BacktestResult with computed metrics.
        """
        job_id = f"backtest_{uuid.uuid4().hex[:12]}"
        logger.info(f"Starting backtest {job_id} for ruleset {ruleset.version}")

        try:
            # Fetch historical features
            features_list = self._fetch_historical_features(start_date, end_date)

            if not features_list:
                logger.warning(
                    f"No features found in date range {start_date} to {end_date}"
                )
                return BacktestResult(
                    job_id=job_id,
                    rule_id=rule_id,
                    ruleset_version=ruleset.version,
                    start_date=start_date,
                    end_date=end_date,
                    metrics=BacktestMetrics(
                        total_records=0,
                        matched_count=0,
                        match_rate=0.0,
                        score_distribution={},
                        score_mean=0.0,
                        score_std=0.0,
                        score_min=0,
                        score_max=0,
                        rejected_count=0,
                        rejected_rate=0.0,
                    ),
                    completed_at=datetime.now(timezone.utc),
                )

            # Replay features through rules
            scores = []
            matched_counts = []
            rejected_counts = []

            for feature_dict in features_list:
                result = evaluate_rules(feature_dict, base_score, ruleset)
                scores.append(result.final_score)
                matched_counts.append(len(result.matched_rules))
                rejected_counts.append(1 if result.rejected else 0)

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

            logger.info(
                f"Backtest {job_id} completed: {metrics.match_rate:.2%} match rate, "
                f"{metrics.rejected_rate:.2%} rejection rate"
            )

            return result

        except Exception as e:
            logger.error(f"Backtest {job_id} failed: {e}", exc_info=True)
            return BacktestResult(
                job_id=job_id,
                rule_id=rule_id,
                ruleset_version=ruleset.version,
                start_date=start_date,
                end_date=end_date,
                metrics=BacktestMetrics(
                    total_records=0,
                    matched_count=0,
                    match_rate=0.0,
                    score_distribution={},
                    score_mean=0.0,
                    score_std=0.0,
                    score_min=0,
                    score_max=0,
                    rejected_count=0,
                    rejected_rate=0.0,
                ),
                completed_at=datetime.now(timezone.utc),
                error=str(e),
            )

    def _fetch_historical_features(
        self, start_date: datetime, end_date: datetime
    ) -> list[dict[str, Any]]:
        """Fetch historical features from database.

        Args:
            start_date: Start of date range.
            end_date: End of date range.

        Returns:
            List of feature dictionaries.
        """
        features_list = []

        with self.db_session.get_session() as session:
            query = text("""
                SELECT
                    velocity_24h,
                    amount_to_avg_ratio_30d,
                    balance_volatility_z_score,
                    experimental_signals
                FROM feature_snapshots
                WHERE computed_at >= :start_date
                  AND computed_at <= :end_date
                ORDER BY computed_at
            """)

            result = session.execute(
                query, {"start_date": start_date, "end_date": end_date}
            )

            for row in result:
                feature_dict = {
                    "velocity_24h": row.velocity_24h,
                    "amount_to_avg_ratio_30d": float(row.amount_to_avg_ratio_30d),
                    "balance_volatility_z_score": float(row.balance_volatility_z_score),
                }

                # Add experimental signals if available
                if row.experimental_signals:
                    exp_signals = row.experimental_signals
                    if isinstance(exp_signals, dict):
                        # Extract common experimental features
                        if "bank_connections_24h" in exp_signals:
                            feature_dict["bank_connections_24h"] = exp_signals[
                                "bank_connections_24h"
                            ]
                        if "merchant_risk_score" in exp_signals:
                            feature_dict["merchant_risk_score"] = exp_signals[
                                "merchant_risk_score"
                            ]
                        if "has_history" in exp_signals:
                            feature_dict["has_history"] = exp_signals["has_history"]

                features_list.append(feature_dict)

        return features_list

    def _compute_metrics(
        self,
        total_records: int,
        scores: list[int],
        matched_counts: list[int],
        rejected_counts: list[int],
    ) -> BacktestMetrics:
        """Compute backtest metrics.

        Args:
            total_records: Total number of records processed.
            scores: List of final scores.
            matched_counts: List of matched rule counts per record.
            rejected_counts: List of rejection flags (0 or 1).

        Returns:
            BacktestMetrics with computed statistics.
        """
        if total_records == 0:
            return BacktestMetrics(
                total_records=0,
                matched_count=0,
                match_rate=0.0,
                score_distribution={},
                score_mean=0.0,
                score_std=0.0,
                score_min=0,
                score_max=0,
                rejected_count=0,
                rejected_rate=0.0,
            )

        scores_array = np.array(scores)
        matched_count = sum(1 for c in matched_counts if c > 0)
        rejected_count = sum(rejected_counts)

        # Score distribution by ranges
        score_ranges = {
            "1-20": 0,
            "21-40": 0,
            "41-60": 0,
            "61-80": 0,
            "81-99": 0,
        }

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
            match_rate=matched_count / total_records if total_records > 0 else 0.0,
            score_distribution=score_ranges,
            score_mean=float(np.mean(scores_array)),
            score_std=float(np.std(scores_array)),
            score_min=int(np.min(scores_array)),
            score_max=int(np.max(scores_array)),
            rejected_count=rejected_count,
            rejected_rate=rejected_count / total_records if total_records > 0 else 0.0,
        )


class BacktestStore:
    """Store and retrieve backtest results."""

    def __init__(self, storage_path: Path | str | None = None):
        """Initialize backtest store.

        Args:
            storage_path: Path for persistent storage. If None, in-memory only.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self._results: dict[str, BacktestResult] = {}

        # Load existing results if file exists
        if self.storage_path and self.storage_path.exists():
            self._load_results()

    def _load_results(self) -> None:
        """Load results from storage file."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)
                self._results = {
                    job_id: BacktestResult.from_dict(result_dict)
                    for job_id, result_dict in data.items()
                }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                f"Failed to load backtest results from {self.storage_path}: {e}"
            )
            self._results = {}

    def _save_results(self) -> None:
        """Save results to storage file."""
        if not self.storage_path:
            return

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                job_id: result.to_dict() for job_id, result in self._results.items()
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save backtest results to {self.storage_path}: {e}")

    def save(self, result: BacktestResult) -> None:
        """Save a backtest result.

        Args:
            result: BacktestResult to save.
        """
        self._results[result.job_id] = result
        self._save_results()

    def get(self, job_id: str) -> BacktestResult | None:
        """Get a backtest result by job ID.

        Args:
            job_id: Job ID.

        Returns:
            BacktestResult if found, None otherwise.
        """
        return self._results.get(job_id)

    def list_results(
        self,
        rule_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[BacktestResult]:
        """List backtest results with optional filters.

        Args:
            rule_id: Filter by rule ID.
            start_date: Filter results after this date.
            end_date: Filter results before this date.

        Returns:
            List of matching BacktestResults, ordered by completed_at (newest first).
        """
        results = list(self._results.values())

        # Apply filters
        if rule_id is not None:
            results = [r for r in results if r.rule_id == rule_id]

        if start_date is not None:
            results = [r for r in results if r.completed_at >= start_date]

        if end_date is not None:
            results = [r for r in results if r.completed_at <= end_date]

        # Sort by completed_at (newest first)
        results.sort(key=lambda r: r.completed_at, reverse=True)

        return results


class BacktestComparator:
    """Compare backtest results and compute deltas."""

    def compute_delta(
        self, base: BacktestMetrics, candidate: BacktestMetrics
    ) -> BacktestDelta:
        """Compute delta between two backtest metrics.

        Deltas are calculated as (candidate - base).

        Args:
            base: Baseline metrics.
            candidate: Candidate metrics.

        Returns:
            BacktestDelta with computed differences.
        """
        return BacktestDelta(
            match_rate_delta=candidate.match_rate - base.match_rate,
            rejected_rate_delta=candidate.rejected_rate - base.rejected_rate,
            score_mean_delta=candidate.score_mean - base.score_mean,
            score_std_delta=candidate.score_std - base.score_std,
            matched_count_delta=candidate.matched_count - base.matched_count,
            rejected_count_delta=candidate.rejected_count - base.rejected_count,
        )


# Global backtest store instance
_global_backtest_store: BacktestStore | None = None


def get_backtest_store() -> BacktestStore:
    """Get the global backtest store instance.

    Returns:
        Global BacktestStore instance.
    """
    global _global_backtest_store
    if _global_backtest_store is None:
        storage_path = os.getenv("BACKTEST_STORAGE_PATH")
        _global_backtest_store = BacktestStore(storage_path=storage_path)
    return _global_backtest_store
