"""Structured logging for inference events."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RuleImpact:
    """Impact of a single rule on the score."""

    rule_id: str
    is_shadow: bool
    score_delta: float
    details: dict[str, Any] | None = None


@dataclass
class InferenceEvent:
    """Structured record of an inference event."""

    request_id: str
    timestamp: datetime
    model_version: str
    rules_version: str
    model_score: int
    final_score: int
    rule_impacts: list[RuleImpact]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class InferenceLogger:
    """Logger for structured inference events.

    For MVP, writes to a rotating JSONL file.
    In production, this would send events to OLAP (BigQuery/ClickHouse).
    """

    def __init__(self, log_path: Path | str = "data/inference_events.jsonl"):
        """Initialize logger.

        Args:
            log_path: Path to log file.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: InferenceEvent) -> None:
        """Log an event.

        Args:
            event: The event to log.
        """
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to log inference event: {e}")

    def query_events(
        self,
        start_time: datetime,
        end_time: datetime,
        rule_id: str | None = None
    ) -> list[InferenceEvent]:
        """Query events from log (Scan-based, inefficient for large datasets)."""
        events = []
        if not self.log_path.exists():
            return events

        try:
            with open(self.log_path) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        ts = datetime.fromisoformat(data["timestamp"])

                        if start_time <= ts <= end_time:
                            # Filter by rule_id if requested
                            if rule_id:
                                impacts = data.get("rule_impacts", [])
                                matched = any(r["rule_id"] == rule_id for r in impacts)
                                if not matched:
                                    continue

                            # Parse back to object
                            # (Simplified parsing)
                            events.append(data)
                    except (ValueError, json.JSONDecodeError):
                        continue
        except Exception as e:
            logger.error(f"Failed to query events: {e}")

        return events # Returning dicts for now to simplify
