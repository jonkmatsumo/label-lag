"""Simple TTL cache for drift detection results."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Module-level singleton
_drift_cache: "DriftCache | None" = None


def _load_cache_ttl() -> int:
    """Load cache TTL from config file.

    Returns:
        TTL in seconds (default: 300).
    """
    config_path = (
        Path(__file__).parent.parent.parent / "config" / "model_thresholds.json"
    )
    default_ttl = 300

    try:
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                drift_config = config.get("drift_thresholds", {})
                return drift_config.get("cache_ttl_seconds", default_ttl)
    except Exception:
        pass

    return default_ttl


@dataclass
class CachedResult:
    """Cached drift detection result."""

    result: dict[str, Any]
    computed_at: datetime
    cache_key: tuple[int, float]


class DriftCache:
    """In-memory cache with TTL for drift results."""

    def __init__(self, ttl_seconds: int = 300):
        """Initialize cache with TTL.

        Args:
            ttl_seconds: Time-to-live in seconds (default: 300 = 5 minutes).
        """
        self._cache: CachedResult | None = None
        self._ttl_seconds = ttl_seconds

    def get(self, hours: int, threshold: float) -> dict[str, Any] | None:
        """Get cached result if valid.

        Args:
            hours: Hours of live data analyzed.
            threshold: PSI threshold used.

        Returns:
            Cached result dict if valid, None otherwise.
        """
        if self._cache is None:
            return None

        # Check if cache key matches
        cache_key = (hours, threshold)
        if self._cache.cache_key != cache_key:
            return None

        # Check if cache is still valid (within TTL)
        now = datetime.now(timezone.utc)
        age_seconds = (now - self._cache.computed_at).total_seconds()
        if age_seconds > self._ttl_seconds:
            return None

        return self._cache.result

    def set(self, hours: int, threshold: float, result: dict[str, Any]) -> None:
        """Store result in cache.

        Args:
            hours: Hours of live data analyzed.
            threshold: PSI threshold used.
            result: Drift detection result dict.
        """
        cache_key = (hours, threshold)
        self._cache = CachedResult(
            result=result,
            computed_at=datetime.now(timezone.utc),
            cache_key=cache_key,
        )

    def invalidate(self) -> None:
        """Clear the cache."""
        self._cache = None


def get_drift_cache() -> DriftCache:
    """Get or create the drift cache singleton.

    Returns:
        DriftCache instance.
    """
    global _drift_cache
    if _drift_cache is None:
        ttl = _load_cache_ttl()
        _drift_cache = DriftCache(ttl_seconds=ttl)
    return _drift_cache
