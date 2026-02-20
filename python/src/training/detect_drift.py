"""Drift detection monitor using Population Stability Index (PSI).

Data is loaded via the Analytics service.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training.crud_client import get_crud_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Model registry name
MODEL_NAME = "ach-fraud-detection"

# Features to monitor for drift
MONITORED_FEATURES = [
    "velocity_24h",
    "amount_to_avg_ratio_30d",
    "balance_volatility_z_score",
]


def _load_drift_thresholds() -> dict[str, float]:
    """Load drift thresholds from config file."""
    config_path = Path(__file__).parents[2] / "config" / "model_thresholds.json"
    # C1: thresholds from env vars
    default_thresholds = {
        "psi_warning": float(os.getenv("DRIFT_PSI_WARN_THRESHOLD", 0.1)),
        "psi_critical": float(os.getenv("DRIFT_PSI_CRIT_THRESHOLD", 0.25)),
        "cache_ttl_seconds": 300,
    }

    try:
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                drift_config = config.get("drift_thresholds", {})
                return {
                    "psi_warning": float(
                        os.getenv(
                            "DRIFT_PSI_WARN_THRESHOLD",
                            drift_config.get(
                                "psi_warning", default_thresholds["psi_warning"]
                            ),
                        )
                    ),
                    "psi_critical": float(
                        os.getenv(
                            "DRIFT_PSI_CRIT_THRESHOLD",
                            drift_config.get(
                                "psi_critical", default_thresholds["psi_critical"]
                            ),
                        )
                    ),
                    "cache_ttl_seconds": drift_config.get(
                        "cache_ttl_seconds",
                        default_thresholds["cache_ttl_seconds"],
                    ),
                }
    except Exception as e:
        logger.warning(
            f"Failed to load drift thresholds from config: {e}. Using defaults."
        )
        return default_thresholds

    return default_thresholds


# Load thresholds from config
_DRIFT_THRESHOLDS = _load_drift_thresholds()
PSI_THRESHOLD_WARNING = _DRIFT_THRESHOLDS["psi_warning"]
PSI_THRESHOLD_CRITICAL = _DRIFT_THRESHOLDS["psi_critical"]
CACHE_TTL_SECONDS = _DRIFT_THRESHOLDS["cache_ttl_seconds"]

MIN_REFERENCE_SAMPLES = 500


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckettype: str = "bins",
    buckets: int = 10,
) -> tuple[float, dict[str, Any]]:
    """Calculate Population Stability Index (PSI) between two distributions."""
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        logger.warning("Empty array provided for PSI calculation")
        return 0.0, {}

    if len(np.unique(expected)) < 2:
        logger.warning(
            "Expected distribution is constant; "
            "skipping PSI calculation and returning 0.0"
        )
        return 0.0, {}

    # Create bucket boundaries
    fallback_reason = None
    if buckettype == "bins":
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        breakpoints = np.linspace(min_val, max_val, buckets + 1)
    elif buckettype == "quantiles":
        quantile_breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        unique_breakpoints = np.unique(quantile_breakpoints)
        if len(unique_breakpoints) < (buckets + 1):
            fallback_reason = "tied_quantiles"
            logger.warning(
                "Quantile PSI bucketing collapsed (%s < %s unique breakpoints). "
                "Falling back to uniform bins over data range.",
                len(unique_breakpoints),
                buckets + 1,
            )
            data_min = min(expected.min(), actual.min())
            data_max = max(expected.max(), actual.max())
            if data_min == data_max:
                return 0.0, {}
            breakpoints = np.linspace(data_min, data_max, buckets + 1)
        else:
            breakpoints = unique_breakpoints
    else:
        raise ValueError(f"Unknown buckettype: {buckettype}")

    # Use range in np.histogram to ensure inclusivity of edges
    hist_kwargs = {}
    if buckettype == "bins" or fallback_reason:
        hist_kwargs = {"bins": buckets, "range": (breakpoints[0], breakpoints[-1])}
    else:
        hist_kwargs = {"bins": breakpoints}

    expected_counts = np.histogram(expected, **hist_kwargs)[0]
    actual_counts = np.histogram(actual, **hist_kwargs)[0]

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    epsilon = 1e-6
    expected_pct = np.clip(expected_pct, epsilon, 1)
    actual_pct = np.clip(actual_pct, epsilon, 1)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = np.sum(psi_values)

    metadata = {
        "buckettype_requested": buckettype,
        "buckettype_used": (
            "bins" if (buckettype == "bins" or fallback_reason) else "quantiles"
        ),
        "buckets_requested": buckets,
        "buckets_used": len(breakpoints) - 1,
        "bucketing_fallback_reason": fallback_reason,
        "breakpoints": [float(b) for b in breakpoints],
        "reference_sample_size": len(expected),
    }

    return float(psi), metadata


def get_reference_data() -> pd.DataFrame | None:
    """Load reference data from MLflow model artifacts."""
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        production_version = None

        for v in versions:
            if v.current_stage == "Production":
                production_version = v
                break

        if production_version is None:
            logger.error("No production model found in registry")
            return None

        run_id = production_version.run_id
        logger.info(
            "Found production model: version=%s, run_id=%s",
            production_version.version,
            run_id,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="reference_data.parquet",
                dst_path=tmpdir,
            )
            df_reference = pd.read_parquet(artifact_path)

        logger.info(f"Loaded reference data: {len(df_reference)} records")
        return df_reference

    except Exception as e:
        logger.error(f"Failed to load reference data: {e}")
        return None


def get_live_data(hours: int = 24) -> pd.DataFrame:
    """Load live data via Analytics service."""
    from analytics.v1 import analytics_pb2

    client = get_crud_client()
    try:
        resp = client.stub.GetDriftWindow(
            analytics_pb2.GetDriftWindowRequest(hours=hours),
            timeout=client.timeout_seconds,
        )
        data = []
        for r in resp.transactions:
            data.append(
                {
                    "velocity_24h": r.velocity_24h,
                    "amount_to_avg_ratio_30d": r.amount_to_avg_ratio_30d,
                    "balance_volatility_z_score": r.balance_volatility_z_score,
                    "computed_at": r.created_at.ToDatetime(),
                }
            )
        df_current = pd.DataFrame(data)
        logger.info(
            "Loaded live data: %s records from last %sh via Analytics",
            len(df_current),
            hours,
        )
        return df_current

    except Exception as e:
        logger.error(f"Failed to load live data from Analytics: {e}")
        return pd.DataFrame()


def detect_drift(
    hours: int = 24,
    threshold: float = PSI_THRESHOLD_CRITICAL,
) -> dict[str, Any]:
    """Run drift detection."""
    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hours_analyzed": hours,
        "threshold": threshold,
        "reference_size": 0,
        "live_size": 0,
        "features": {},
        "drift_detected": False,
        "drifted_features": [],
        "alerts": [],
    }

    df_reference = get_reference_data()
    if df_reference is None or len(df_reference) == 0:
        results["error"] = "No reference data available"
        return results

    results["reference_size"] = len(df_reference)

    if len(df_reference) < MIN_REFERENCE_SAMPLES:
        msg = (
            f"Insufficient reference data: {len(df_reference)} samples "
            f"(minimum {MIN_REFERENCE_SAMPLES})"
        )
        logger.warning(msg)
        results["error"] = msg
        return results

    df_current = get_live_data(hours=hours)
    if len(df_current) == 0:
        results["error"] = "No live data available"
        return results

    results["live_size"] = len(df_current)

    for feature in MONITORED_FEATURES:
        if feature not in df_reference.columns or feature not in df_current.columns:
            continue

        expected = df_reference[feature].values.astype(float)
        actual = df_current[feature].values.astype(float)

        psi, bucketing = calculate_psi(
            expected, actual, buckettype="quantiles", buckets=10
        )

        if psi >= PSI_THRESHOLD_CRITICAL:
            status = "CRITICAL"
            results["alerts"].append(
                {
                    "severity": "critical",
                    "feature": feature,
                    "psi": round(psi, 4),
                    "threshold": PSI_THRESHOLD_CRITICAL,
                    "recommendation": (
                        f"Feature '{feature}' shows critical drift "
                        f"(PSI={psi:.4f}). Immediate model retraining or "
                        "data quality audit recommended."
                    ),
                }
            )
        elif psi >= PSI_THRESHOLD_WARNING:
            status = "WARNING"
            results["alerts"].append(
                {
                    "severity": "warning",
                    "feature": feature,
                    "psi": round(psi, 4),
                    "threshold": PSI_THRESHOLD_WARNING,
                    "recommendation": (
                        f"Feature '{feature}' shows moderate drift "
                        f"(PSI={psi:.4f}). Monitor closely and consider "
                        "retraining if trend continues."
                    ),
                }
            )
        else:
            status = "OK"

        results["features"][feature] = {
            "psi": round(psi, 4),
            "status": status,
            "bucketing": bucketing,
        }

        if status == "CRITICAL":
            results["drift_detected"] = True
            results["drifted_features"].append(feature)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect feature drift")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=PSI_THRESHOLD_CRITICAL)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    results = detect_drift(hours=args.hours, threshold=args.threshold)

    if args.json:
        print(json.dumps(results, indent=2))

    if "error" in results:
        return 2
    if results["drift_detected"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
