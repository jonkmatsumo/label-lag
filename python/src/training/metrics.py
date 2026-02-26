"""Training observability metrics."""

from __future__ import annotations

from datetime import UTC, datetime

from prometheus_client import Counter, Histogram

from training.jobs import TuningJob

training_job_cancellations_total = Counter(
    "training_job_cancellations_total",
    "Number of training/tuning jobs that transitioned to canceled.",
)

training_job_cancel_latency_ms = Histogram(
    "training_job_cancel_latency_ms",
    "Latency from job start (or creation) until canceled state, in milliseconds.",
    buckets=(50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000, 300000),
)


def observe_training_job_cancellation(
    job: TuningJob, *, canceled_at: datetime | None = None
) -> None:
    """Record cancellation count and cancel latency with bounded dimensions."""
    training_job_cancellations_total.inc()
    anchor_ts = job.started_at or job.created_at
    end_ts = canceled_at or job.ended_at or datetime.now(UTC)
    latency_ms = max(0.0, (end_ts - anchor_ts).total_seconds() * 1000.0)
    training_job_cancel_latency_ms.observe(latency_ms)
