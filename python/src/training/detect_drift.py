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
from training.reason_codes import (
    DRIFT_ERROR_CODES,
    DRIFT_RESOLUTION_MODES,
    DriftErrorCode,
    DriftFallbackReason,
    DriftResolutionMode,
)

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
PSI_MIN_EXPECTED_PER_BUCKET = float(os.getenv("DRIFT_PSI_MIN_EXPECTED_PER_BUCKET", "5"))
PSI_MIN_NONEMPTY_BUCKETS_RATIO = float(
    os.getenv("DRIFT_PSI_MIN_NONEMPTY_BUCKETS_RATIO", "0.6")
)
DRIFT_REFERENCE_MODEL_ALIAS = os.getenv("DRIFT_REFERENCE_MODEL_ALIAS", "").strip()
_DRIFT_STAGE_FALLBACK_WARNED = False
_DRIFT_LATEST_FALLBACK_WARNED = False
MAX_DRIFT_ERROR_MESSAGE_LENGTH = 200
MAX_REFERENCE_METADATA_TEXT_LENGTH = 64
MAX_REFERENCE_RUN_ID_LENGTH = 128
MAX_BREAKPOINTS_COUNT = 20
MAX_DRIFT_REFERENCE_VERSION_LENGTH = MAX_REFERENCE_METADATA_TEXT_LENGTH
MAX_DRIFT_REFERENCE_RUN_ID_LENGTH = MAX_REFERENCE_RUN_ID_LENGTH
MAX_DRIFT_REFERENCE_ALIAS_LENGTH = MAX_REFERENCE_METADATA_TEXT_LENGTH
MAX_DRIFT_BUCKETTYPE_LENGTH = 24
MAX_DRIFT_BREAKPOINTS = MAX_BREAKPOINTS_COUNT
MAX_REFERENCE_RESOLUTION_WARNING_LENGTH = 64
REFERENCE_RESOLUTION_WARNING_CODES = frozenset(
    {
        "alias_not_found_fallback",
        "alias_ambiguous_selected_highest",
        "stage_fallback_used",
        "latest_fallback_used",
        "no_reference_versions_available",
    }
)
_DEFAULT_DRIFT_ERROR_MESSAGES = {
    DriftErrorCode.NO_REFERENCE_DATA.value: "No reference data available",
    DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES.value: (
        "Insufficient reference data"
    ),
    DriftErrorCode.NO_LIVE_DATA.value: "No live data available",
    DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value: (
        "Drift signal suppressed due to insufficient bucket mass"
    ),
}
_LEGACY_DRIFT_ERROR_CODE_ALIASES = {
    "no_reference_model": DriftErrorCode.NO_REFERENCE_DATA.value,
    "no_reference": DriftErrorCode.NO_REFERENCE_DATA.value,
    "insufficient_reference_data": (
        DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES.value
    ),
    "insufficient_reference": DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES.value,
    "no_live_window": DriftErrorCode.NO_LIVE_DATA.value,
    "insufficient_bucket_mass": DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value,
}
_LEGACY_RESOLUTION_MODE_ALIASES = {
    "production_stage": DriftResolutionMode.STAGE.value,
    "latest_version": DriftResolutionMode.LATEST.value,
}
MAX_DRIFT_ERROR_CODE_LENGTH = 64
_REFERENCE_RESOLUTION_METADATA_KEYS = (
    "requested_alias",
    "resolution_strategy",
    "resolution_mode",
    "alias_candidate_count",
    "alias_ambiguous",
    "selected_model_version",
    "selected_run_id",
)
_BUCKETING_METADATA_KEYS = (
    "buckettype_requested",
    "buckettype_used",
    "buckets_requested",
    "buckets_used",
    "bucketing_fallback_reason",
    "breakpoints",
    "reference_sample_size",
    "nonempty_buckets",
    "nonempty_buckets_ratio",
    "min_expected_count",
    "bucket_mass_ok",
    "bucket_mass_guardrail_applied",
    "drift_error",
)


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
            fallback_reason = DriftFallbackReason.TIED_QUANTILES.value
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

    bucket_count = len(expected_counts)
    nonempty_buckets = int(np.count_nonzero(expected_counts > 0))
    nonempty_buckets_ratio = (
        float(nonempty_buckets) / float(bucket_count) if bucket_count > 0 else 0.0
    )
    min_expected_count = float(expected_counts.min()) if bucket_count > 0 else 0.0
    bucket_mass_ok_raw = (
        min_expected_count >= PSI_MIN_EXPECTED_PER_BUCKET
        and nonempty_buckets_ratio >= PSI_MIN_NONEMPTY_BUCKETS_RATIO
    )
    bucket_mass_guardrail_applied = len(expected) >= MIN_REFERENCE_SAMPLES
    bucket_mass_ok = bucket_mass_ok_raw if bucket_mass_guardrail_applied else True

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
        "nonempty_buckets": nonempty_buckets,
        "nonempty_buckets_ratio": round(nonempty_buckets_ratio, 4),
        "min_expected_count": min_expected_count,
        "bucket_mass_ok": bucket_mass_ok,
        "bucket_mass_guardrail_applied": bucket_mass_guardrail_applied,
    }

    if not bucket_mass_ok:
        logger.warning(
            "Insufficient bucket mass for PSI "
            "(nonempty_buckets=%s/%s ratio=%.3f min_expected_count=%.2f "
            "required_nonempty_ratio=%.3f min_expected_per_bucket=%.2f)",
            nonempty_buckets,
            bucket_count,
            nonempty_buckets_ratio,
            min_expected_count,
            PSI_MIN_NONEMPTY_BUCKETS_RATIO,
            PSI_MIN_EXPECTED_PER_BUCKET,
        )
        metadata["drift_error"] = DriftFallbackReason.INSUFFICIENT_BUCKET_MASS.value
        return 0.0, metadata

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    epsilon = 1e-6
    expected_pct = np.clip(expected_pct, epsilon, 1)
    actual_pct = np.clip(actual_pct, epsilon, 1)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = np.sum(psi_values)

    return float(psi), metadata


def _safe_model_version_number(model_version: Any) -> int:
    """Parse model version into an integer for deterministic ordering."""
    try:
        return int(str(getattr(model_version, "version", "")).strip())
    except Exception:
        return -1


def _model_version_has_alias(model_version: Any, alias_name: str) -> bool:
    """Check if a model version contains a given alias."""
    normalized_alias = str(alias_name).lstrip("@")
    aliases = getattr(model_version, "aliases", None)
    if aliases is None:
        return False
    if isinstance(aliases, str):
        return aliases.lstrip("@") == normalized_alias
    try:
        return any(str(alias).lstrip("@") == normalized_alias for alias in aliases)
    except Exception:
        return False


def _select_reference_model_version(
    versions: list[Any],
    *,
    alias_name: str | None,
) -> tuple[Any | None, dict[str, Any]]:
    """Resolve reference model version with a deterministic fallback policy.

    Resolution Order:
    1. Explicit Alias: Match by DRIFT_REFERENCE_MODEL_ALIAS (e.g., 'champion').
       If multiple versions share the same alias, selects the highest version.
    2. Production Stage: Fall back to versions in the 'Production' stage.
       If multiple exist, selects the highest version. (Deprecated: Warns once).
    3. Latest Version: Final fallback to the highest registered version number.
       (Warns once).
    """
    metadata: dict[str, Any] = {
        "requested_alias": None,
        "requested_mode": DriftResolutionMode.STAGE.value,
        "resolution_strategy": None,
        "resolution_warning": None,
        "alias_candidate_count": 0,
        "alias_ambiguous": False,
    }

    if not versions:
        metadata["resolution_warning"] = "no_reference_versions_available"
        return None, metadata

    normalized_alias = str(alias_name or "").strip().lstrip("@")
    if normalized_alias:
        metadata["requested_alias"] = normalized_alias
        metadata["requested_mode"] = DriftResolutionMode.ALIAS.value
        alias_candidates = [
            version
            for version in versions
            if _model_version_has_alias(version, normalized_alias)
        ]
        if alias_candidates:
            alias_candidates_sorted = sorted(
                alias_candidates,
                key=_safe_model_version_number,
                reverse=True,
            )
            metadata["alias_candidate_count"] = len(alias_candidates_sorted)
            metadata["alias_ambiguous"] = len(alias_candidates_sorted) > 1
            selected_alias_version = alias_candidates_sorted[0]
            if metadata["alias_ambiguous"]:
                metadata["resolution_warning"] = "alias_ambiguous_selected_highest"
                logger.warning(
                    "Drift alias '@%s' resolved to %s candidates; selecting highest "
                    "version deterministically (%s).",
                    normalized_alias,
                    len(alias_candidates_sorted),
                    getattr(selected_alias_version, "version", "unknown"),
                )
            metadata["resolution_strategy"] = "alias"
            return selected_alias_version, metadata
        logger.warning(
            "Configured drift reference alias '@%s' not found; "
            "falling back to stage/latest resolution.",
            normalized_alias,
        )
        metadata["resolution_warning"] = "alias_not_found_fallback"

    stage_candidates = [
        version
        for version in versions
        if getattr(version, "current_stage", "") == "Production"
    ]
    if stage_candidates:
        selected_stage_version = sorted(
            stage_candidates, key=_safe_model_version_number, reverse=True
        )[0]
        metadata["resolution_strategy"] = "production_stage"
        if metadata.get("resolution_warning") is None:
            metadata["resolution_warning"] = "stage_fallback_used"

        global _DRIFT_STAGE_FALLBACK_WARNED
        if not _DRIFT_STAGE_FALLBACK_WARNED:
            logger.warning(
                "Using legacy Production stage for drift reference resolution. "
                "Set DRIFT_REFERENCE_MODEL_ALIAS for deterministic "
                "alias-based selection."
            )
            _DRIFT_STAGE_FALLBACK_WARNED = True

        return selected_stage_version, metadata

    selected_latest = sorted(versions, key=_safe_model_version_number, reverse=True)[0]
    metadata["resolution_strategy"] = "latest_version"
    if metadata.get("resolution_warning") is None:
        metadata["resolution_warning"] = "latest_fallback_used"

    global _DRIFT_LATEST_FALLBACK_WARNED
    if not _DRIFT_LATEST_FALLBACK_WARNED:
        logger.warning(
            "No alias/stage reference found for drift; "
            "falling back to latest model version %s.",
            getattr(selected_latest, "version", "unknown"),
        )
        _DRIFT_LATEST_FALLBACK_WARNED = True

    return selected_latest, metadata


def _normalize_resolution_mode(raw_strategy: Any) -> str:
    if raw_strategy is None:
        return DriftResolutionMode.NONE.value
    normalized = str(raw_strategy).strip().lower()
    if normalized in DRIFT_RESOLUTION_MODES:
        return normalized
    if normalized in _LEGACY_RESOLUTION_MODE_ALIASES:
        return _LEGACY_RESOLUTION_MODE_ALIASES[normalized]
    return DriftResolutionMode.NONE.value


def _normalize_requested_resolution_mode(
    raw_mode: Any,
    *,
    requested_alias: Any,
) -> str:
    normalized = _normalize_resolution_mode(raw_mode)
    if normalized != DriftResolutionMode.NONE.value:
        return normalized
    alias = _bounded_optional_text(
        requested_alias,
        max_len=MAX_REFERENCE_METADATA_TEXT_LENGTH,
    )
    if alias is not None:
        return DriftResolutionMode.ALIAS.value
    return DriftResolutionMode.STAGE.value


def _normalize_reference_resolution_warning(raw_warning: Any) -> str | None:
    warning = _bounded_optional_text(
        raw_warning,
        max_len=MAX_REFERENCE_RESOLUTION_WARNING_LENGTH,
    )
    if warning is None:
        return None
    warning = warning.lower()
    if warning not in REFERENCE_RESOLUTION_WARNING_CODES:
        return None
    return warning


def _bounded_metadata_text(value: Any, *, max_len: int) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    return rendered[:max_len]


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
        if np.isfinite(parsed):
            return parsed
        return None
    except (TypeError, ValueError):
        return None


def _normalize_reference_resolution(raw_resolution: Any) -> dict[str, Any]:
    source = raw_resolution if isinstance(raw_resolution, dict) else {}
    requested_alias = _bounded_metadata_text(
        source.get("requested_alias") or DRIFT_REFERENCE_MODEL_ALIAS,
        max_len=MAX_DRIFT_REFERENCE_ALIAS_LENGTH,
    )
    normalized_mode = _normalize_resolution_mode(
        source.get("resolution_mode") or source.get("resolution_strategy")
    )
    alias_candidate_count = _coerce_int_or_none(source.get("alias_candidate_count"))
    if alias_candidate_count is None or alias_candidate_count < 0:
        alias_candidate_count = 0

    alias_ambiguous_raw = source.get("alias_ambiguous")
    alias_ambiguous = (
        bool(alias_ambiguous_raw) if alias_ambiguous_raw is not None else False
    )

    return {
        "requested_alias": requested_alias,
        "resolution_strategy": normalized_mode,
        "resolution_mode": normalized_mode,
        "alias_candidate_count": alias_candidate_count,
        "alias_ambiguous": alias_ambiguous,
        "selected_model_version": _bounded_metadata_text(
            source.get("selected_model_version"),
            max_len=MAX_DRIFT_REFERENCE_VERSION_LENGTH,
        ),
        "selected_run_id": _bounded_metadata_text(
            source.get("selected_run_id"),
            max_len=MAX_DRIFT_REFERENCE_RUN_ID_LENGTH,
        ),
    }


def _bounded_error_message(message: Any) -> str:
    if message is None:
        return ""
    return str(message).strip()[:MAX_DRIFT_ERROR_MESSAGE_LENGTH]


def _bounded_error_code(code: Any) -> str | None:
    if code is None:
        return None
    rendered = str(code).strip()
    if not rendered:
        return None
    return rendered[:MAX_DRIFT_ERROR_CODE_LENGTH]


def _bounded_optional_text(value: Any, *, max_len: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_len]


def _normalize_reference_resolution_metadata(
    raw_resolution: Any,
    *,
    fallback_mode: str,
) -> dict[str, Any]:
    metadata = raw_resolution if isinstance(raw_resolution, dict) else {}
    raw_resolution_strategy = _bounded_optional_text(
        metadata.get("resolution_strategy"),
        max_len=MAX_REFERENCE_METADATA_TEXT_LENGTH,
    )
    canonical_resolution_mode = _normalize_resolution_mode(
        metadata.get("resolution_mode") or raw_resolution_strategy or fallback_mode
    )
    alias_candidate_count = _coerce_int_or_none(metadata.get("alias_candidate_count"))
    if alias_candidate_count is None or alias_candidate_count < 0:
        alias_candidate_count = 0

    normalized = {
        "requested_alias": _bounded_optional_text(
            metadata.get("requested_alias") or DRIFT_REFERENCE_MODEL_ALIAS,
            max_len=MAX_REFERENCE_METADATA_TEXT_LENGTH,
        ),
        "resolution_strategy": raw_resolution_strategy or canonical_resolution_mode,
        "resolution_mode": canonical_resolution_mode,
        "alias_candidate_count": alias_candidate_count,
        "alias_ambiguous": bool(metadata.get("alias_ambiguous", False)),
        "selected_model_version": _bounded_optional_text(
            metadata.get("selected_model_version"),
            max_len=MAX_REFERENCE_METADATA_TEXT_LENGTH,
        ),
        "selected_run_id": _bounded_optional_text(
            metadata.get("selected_run_id"),
            max_len=MAX_REFERENCE_RUN_ID_LENGTH,
        ),
    }
    for key in _REFERENCE_RESOLUTION_METADATA_KEYS:
        normalized.setdefault(key, None)
    return normalized


def _normalize_bucketing_metadata(raw_bucketing: Any) -> dict[str, Any]:
    metadata = raw_bucketing if isinstance(raw_bucketing, dict) else {}
    normalized_breakpoints: list[float] = []
    raw_breakpoints = metadata.get("breakpoints")
    if isinstance(raw_breakpoints, list | tuple):
        for value in raw_breakpoints:
            parsed = _coerce_float_or_none(value)
            if parsed is not None:
                normalized_breakpoints.append(parsed)
            if len(normalized_breakpoints) >= MAX_BREAKPOINTS_COUNT:
                break

    drift_error = _normalize_error_code(metadata.get("drift_error"))
    if drift_error is None:
        drift_error = _bounded_error_code(metadata.get("drift_error"))

    normalized = {
        "buckettype_requested": _bounded_optional_text(
            metadata.get("buckettype_requested"),
            max_len=MAX_DRIFT_BUCKETTYPE_LENGTH,
        ),
        "buckettype_used": _bounded_optional_text(
            metadata.get("buckettype_used"),
            max_len=MAX_DRIFT_BUCKETTYPE_LENGTH,
        ),
        "buckets_requested": _coerce_int_or_none(metadata.get("buckets_requested")),
        "buckets_used": _coerce_int_or_none(metadata.get("buckets_used")),
        "bucketing_fallback_reason": _bounded_optional_text(
            metadata.get("bucketing_fallback_reason"),
            max_len=MAX_DRIFT_ERROR_CODE_LENGTH,
        ),
        "breakpoints": normalized_breakpoints,
        "reference_sample_size": _coerce_int_or_none(
            metadata.get("reference_sample_size")
        ),
        "nonempty_buckets": _coerce_int_or_none(metadata.get("nonempty_buckets")),
        "nonempty_buckets_ratio": _coerce_float_or_none(
            metadata.get("nonempty_buckets_ratio")
        ),
        "min_expected_count": _coerce_float_or_none(metadata.get("min_expected_count")),
        "bucket_mass_ok": (
            bool(metadata["bucket_mass_ok"]) if "bucket_mass_ok" in metadata else None
        ),
        "bucket_mass_guardrail_applied": (
            bool(metadata["bucket_mass_guardrail_applied"])
            if "bucket_mass_guardrail_applied" in metadata
            else None
        ),
        "drift_error": drift_error,
    }
    for key in _BUCKETING_METADATA_KEYS:
        normalized.setdefault(key, None)
    return normalized


def _set_canonical_error(
    results: dict[str, Any],
    *,
    code: DriftErrorCode,
    message: Any,
) -> None:
    bounded_message = _bounded_error_message(message)
    bounded_code = _bounded_error_code(code.value) or code.value
    results["error"] = bounded_message
    results["error_code"] = bounded_code
    results["error_message"] = bounded_message
    if results.get("drift_error") is None:
        results["drift_error"] = bounded_code


def _normalize_error_code(raw_code: Any) -> str | None:
    if raw_code is None:
        return None
    normalized = str(raw_code).strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if lowered in DRIFT_ERROR_CODES:
        return lowered
    if lowered in _LEGACY_DRIFT_ERROR_CODE_ALIASES:
        return _LEGACY_DRIFT_ERROR_CODE_ALIASES[lowered]
    if lowered == DriftFallbackReason.INSUFFICIENT_BUCKET_MASS.value:
        return DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value
    return None


def _infer_error_code_from_message(message: Any) -> str | None:
    if message is None:
        return None
    lowered = str(message).strip().lower()
    if not lowered:
        return None
    if "insufficient reference data" in lowered:
        return DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES.value
    if "reference data" in lowered:
        return DriftErrorCode.NO_REFERENCE_DATA.value
    if "live data" in lowered:
        return DriftErrorCode.NO_LIVE_DATA.value
    if "bucket mass" in lowered:
        return DriftErrorCode.INSUFFICIENT_BUCKET_MASS.value
    return None


def _finalize_drift_error_contract(results: dict[str, Any]) -> dict[str, Any]:
    """Normalize additive drift error contract fields for stability."""
    results.setdefault("error_code", None)
    results.setdefault("error_message", None)
    results.setdefault("drift_error", None)
    results.setdefault("reference_resolution", {})
    results.setdefault("reference_model_version", None)
    results.setdefault("reference_resolution_mode_requested", None)
    results.setdefault("reference_resolution_mode", None)
    results.setdefault("reference_model_version_chosen", None)
    results.setdefault("reference_alias_requested", None)
    results.setdefault("reference_resolution_warning", None)
    raw_resolution_mode = _normalize_resolution_mode(results.get("resolution_mode"))
    reference_resolution = _normalize_reference_resolution_metadata(
        results.get("reference_resolution"),
        fallback_mode=raw_resolution_mode,
    )
    normalized_resolution_mode = raw_resolution_mode
    if normalized_resolution_mode == DriftResolutionMode.NONE.value:
        normalized_resolution_mode = _normalize_resolution_mode(
            reference_resolution.get("resolution_mode")
            or reference_resolution.get("resolution_strategy")
        )
    reference_resolution["resolution_strategy"] = normalized_resolution_mode
    reference_resolution["resolution_mode"] = normalized_resolution_mode
    results["reference_resolution"] = reference_resolution
    results["resolution_mode"] = normalized_resolution_mode

    error_code = _normalize_error_code(results.get("error_code"))
    if error_code is None:
        error_code = _normalize_error_code(results.get("drift_error"))
    if error_code is None:
        error_code = _infer_error_code_from_message(results.get("error"))

    error_message = results.get("error_message")
    if error_message is None:
        error_message = results.get("error")
    if error_message is None and error_code is not None:
        error_message = _DEFAULT_DRIFT_ERROR_MESSAGES.get(error_code)
    if error_message is not None:
        bounded_error_message = _bounded_error_message(error_message)
        if bounded_error_message:
            results["error_message"] = bounded_error_message
            results["error"] = bounded_error_message
        else:
            results["error_message"] = None
            results["error"] = None
    else:
        results["error_message"] = None
        results["error"] = None

    if error_code is not None:
        results["drift_error"] = error_code
    else:
        results["drift_error"] = _normalize_error_code(results.get("drift_error"))
    if results["drift_error"] is None:
        results["drift_error"] = _bounded_error_code(results.get("drift_error"))

    results["error_code"] = error_code

    reference_model_version = _bounded_optional_text(
        results.get("reference_model_version"),
        max_len=MAX_REFERENCE_METADATA_TEXT_LENGTH,
    )
    if reference_model_version is None:
        reference_model_version = _bounded_optional_text(
            reference_resolution.get("selected_model_version"),
            max_len=MAX_REFERENCE_METADATA_TEXT_LENGTH,
        )
    results["reference_model_version"] = reference_model_version
    if reference_resolution.get("selected_model_version") is None:
        reference_resolution["selected_model_version"] = reference_model_version

    requested_alias = _bounded_optional_text(
        results.get("reference_alias_requested"),
        max_len=MAX_REFERENCE_METADATA_TEXT_LENGTH,
    )
    if requested_alias is None:
        requested_alias = _bounded_optional_text(
            reference_resolution.get("requested_alias"),
            max_len=MAX_REFERENCE_METADATA_TEXT_LENGTH,
        )
    results["reference_alias_requested"] = requested_alias
    requested_mode = _normalize_requested_resolution_mode(
        results.get("reference_resolution_mode_requested"),
        requested_alias=requested_alias,
    )
    results["reference_resolution_mode_requested"] = requested_mode
    results["reference_resolution_mode"] = normalized_resolution_mode
    results["reference_model_version_chosen"] = reference_model_version

    reference_resolution_warning = _normalize_reference_resolution_warning(
        results.get("reference_resolution_warning")
    )
    if (
        reference_resolution_warning is None
        and requested_mode == DriftResolutionMode.ALIAS.value
        and normalized_resolution_mode
        in {DriftResolutionMode.STAGE.value, DriftResolutionMode.LATEST.value}
    ):
        reference_resolution_warning = "alias_not_found_fallback"
    if (
        reference_resolution_warning is None
        and normalized_resolution_mode == DriftResolutionMode.NONE.value
        and reference_model_version is None
    ):
        reference_resolution_warning = "no_reference_versions_available"
    results["reference_resolution_warning"] = reference_resolution_warning

    return results


def get_reference_data(
    *, include_metadata: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame | None, dict[str, Any]] | None:
    """Load reference data from MLflow model artifacts."""
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
        selected_version, resolution_metadata = _select_reference_model_version(
            versions,
            alias_name=DRIFT_REFERENCE_MODEL_ALIAS,
        )

        if selected_version is None:
            logger.error("No registered model versions found for drift reference")
            if include_metadata:
                return None, resolution_metadata
            return None

        run_id = selected_version.run_id
        selected_version_number = str(getattr(selected_version, "version", "unknown"))
        resolution_metadata["selected_model_version"] = selected_version_number
        resolution_metadata["selected_run_id"] = str(run_id)
        logger.info(
            "Resolved drift reference model: strategy=%s version=%s run_id=%s",
            resolution_metadata.get("resolution_strategy"),
            selected_version_number,
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
        if include_metadata:
            return df_reference, resolution_metadata
        return df_reference

    except Exception as e:
        logger.error(f"Failed to load reference data: {e}")
        if include_metadata:
            return None, {}
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
    initial_reference_resolution = _normalize_reference_resolution({})
    initial_requested_alias = _bounded_optional_text(
        DRIFT_REFERENCE_MODEL_ALIAS.lstrip("@"),
        max_len=MAX_REFERENCE_METADATA_TEXT_LENGTH,
    )
    initial_requested_mode = _normalize_requested_resolution_mode(
        None,
        requested_alias=initial_requested_alias,
    )
    results = {
        "timestamp": datetime.now(UTC).isoformat(),
        "hours_analyzed": hours,
        "threshold": threshold,
        "reference_size": 0,
        "live_size": 0,
        "features": {},
        "drift_detected": False,
        "drifted_features": [],
        "drift_error": None,
        "error_code": None,
        "error_message": None,
        "error": None,
        "resolution_mode": DriftResolutionMode.NONE.value,
        "alerts": [],
        "reference_resolution": initial_reference_resolution,
        "reference_model_version": None,
        "reference_resolution_mode_requested": initial_requested_mode,
        "reference_resolution_mode": DriftResolutionMode.NONE.value,
        "reference_model_version_chosen": None,
        "reference_alias_requested": initial_requested_alias,
        "reference_resolution_warning": None,
    }

    reference_result = get_reference_data(include_metadata=True)
    reference_resolution: dict[str, Any] = initial_reference_resolution
    if isinstance(reference_result, tuple):
        df_reference, raw_reference_resolution = reference_result
        if isinstance(raw_reference_resolution, dict):
            results["reference_resolution_mode_requested"] = (
                _normalize_requested_resolution_mode(
                    raw_reference_resolution.get("requested_mode"),
                    requested_alias=raw_reference_resolution.get("requested_alias")
                    or initial_requested_alias,
                )
            )
            results["reference_alias_requested"] = _bounded_optional_text(
                raw_reference_resolution.get("requested_alias")
                or initial_requested_alias,
                max_len=MAX_REFERENCE_METADATA_TEXT_LENGTH,
            )
            results["reference_resolution_warning"] = (
                _normalize_reference_resolution_warning(
                    raw_reference_resolution.get("resolution_warning")
                )
            )
        reference_resolution = _normalize_reference_resolution(raw_reference_resolution)
    else:
        df_reference = reference_result

    results["reference_resolution"] = reference_resolution
    selected_version = reference_resolution.get("selected_model_version")
    if selected_version is not None:
        results["reference_model_version"] = selected_version
    results["resolution_mode"] = _normalize_resolution_mode(
        reference_resolution.get("resolution_mode")
        or reference_resolution.get("resolution_strategy")
    )
    results["reference_resolution_mode"] = results["resolution_mode"]

    if df_reference is None or len(df_reference) == 0:
        _set_canonical_error(
            results,
            code=DriftErrorCode.NO_REFERENCE_DATA,
            message="No reference data available",
        )
        return _finalize_drift_error_contract(results)

    results["reference_size"] = len(df_reference)

    if len(df_reference) < MIN_REFERENCE_SAMPLES:
        msg = (
            f"Insufficient reference data: {len(df_reference)} samples "
            f"(minimum {MIN_REFERENCE_SAMPLES})"
        )
        logger.warning(msg)
        _set_canonical_error(
            results,
            code=DriftErrorCode.INSUFFICIENT_REFERENCE_SAMPLES,
            message=msg,
        )
        return _finalize_drift_error_contract(results)

    df_current = get_live_data(hours=hours)
    if len(df_current) == 0:
        _set_canonical_error(
            results,
            code=DriftErrorCode.NO_LIVE_DATA,
            message="No live data available",
        )
        return _finalize_drift_error_contract(results)

    results["live_size"] = len(df_current)

    for feature in MONITORED_FEATURES:
        if feature not in df_reference.columns or feature not in df_current.columns:
            continue

        expected = df_reference[feature].values.astype(float)
        actual = df_current[feature].values.astype(float)

        psi, bucketing_raw = calculate_psi(
            expected, actual, buckettype="quantiles", buckets=10
        )
        normalized_bucketing = _normalize_bucketing_metadata(bucketing_raw)
        drift_error = _normalize_error_code(normalized_bucketing.get("drift_error"))

        if drift_error == DriftFallbackReason.INSUFFICIENT_BUCKET_MASS.value:
            results["drift_error"] = drift_error
            if results["error_message"] is None:
                _set_canonical_error(
                    results,
                    code=DriftErrorCode.INSUFFICIENT_BUCKET_MASS,
                    message="Drift signal suppressed due to insufficient bucket mass",
                )
            results["features"][feature] = {
                "psi": round(psi, 4),
                "status": "OK",
                "drift_error": drift_error,
                "bucketing": normalized_bucketing,
            }
            continue

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
            "drift_error": drift_error,
            "bucketing": normalized_bucketing,
        }

        if status == "CRITICAL":
            results["drift_detected"] = True
            results["drifted_features"].append(feature)

    return _finalize_drift_error_contract(results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect feature drift")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=PSI_THRESHOLD_CRITICAL)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    results = detect_drift(hours=args.hours, threshold=args.threshold)

    if args.json:
        print(json.dumps(results, indent=2))

    if results.get("error_code") is not None:
        return 2
    if results["drift_detected"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
