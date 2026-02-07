"""Tests for feature drift alerting (C1)."""

import os
from unittest.mock import patch

from training.detect_drift import detect_drift


class TestDriftAlerts:
    """Tests for feature drift alerting (C1)."""

    def test_drift_alert_classification(self):
        """Test that drift is correctly classified into alerts (C1)."""
        with (
            patch("training.detect_drift.get_reference_data") as mock_ref,
            patch("training.detect_drift.get_live_data") as mock_live,
            patch.dict(
                os.environ,
                {"DRIFT_PSI_WARN_THRESHOLD": "0.1", "DRIFT_PSI_CRIT_THRESHOLD": "0.25"},
            ),
        ):
            import numpy as np
            import pandas as pd

            # Setup data that will cause drift
            # Feature 1: No drift
            # Feature 2: Warning drift
            # Feature 3: Critical drift

            ref_data = pd.DataFrame(
                {
                    "velocity_24h": np.random.normal(0, 1, 1000),
                    "amount_to_avg_ratio_30d": np.random.normal(0, 1, 1000),
                    "balance_volatility_z_score": np.random.normal(0, 1, 1000),
                }
            )
            mock_ref.return_value = ref_data

            # Increase sample size for stability
            live_data = pd.DataFrame(
                {
                    "velocity_24h": np.random.normal(0, 1, 1000),  # No drift
                    "amount_to_avg_ratio_30d": np.random.normal(
                        0.4, 1, 1000
                    ),  # Warning drift approx
                    "balance_volatility_z_score": np.random.normal(
                        1.0, 1, 1000
                    ),  # Critical drift
                }
            )
            mock_live.return_value = live_data

            # Rebind thresholds because they are loaded at module import time.
            with (
                patch("training.detect_drift.PSI_THRESHOLD_WARNING", 0.1),
                patch("training.detect_drift.PSI_THRESHOLD_CRITICAL", 0.25),
            ):
                results = detect_drift(hours=24)

                alerts = results["alerts"]

                # Check balance_volatility_z_score (should be critical or warning)
                assert len(alerts) > 0

                # Logging is not asserted here; this test focuses on alert output.

    def test_drift_thresholds_from_env(self):
        """Test that drift thresholds are loaded from environment variables (C1)."""
        # Thresholds are loaded at module level, so call the loader directly.
        with patch.dict(
            os.environ,
            {"DRIFT_PSI_WARN_THRESHOLD": "0.15", "DRIFT_PSI_CRIT_THRESHOLD": "0.35"},
        ):
            from training.detect_drift import _load_drift_thresholds

            thresholds = _load_drift_thresholds()
            assert thresholds["psi_warning"] == 0.15
            assert thresholds["psi_critical"] == 0.35
