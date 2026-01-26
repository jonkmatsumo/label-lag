"""Tests for drift monitoring UI components."""

from unittest.mock import MagicMock, patch

from ui.data_service import fetch_drift_status


class TestFetchDriftStatus:
    """Tests for fetch_drift_status function."""

    @patch("ui.data_service.requests.get")
    def test_success_returns_dict(self, mock_get):
        """API success should return dict."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "ok",
            "computed_at": "2024-01-01T12:00:00+00:00",
            "cached": False,
            "reference_window": "Production model",
            "current_window": "Last 24 hours",
            "reference_size": 1000,
            "live_size": 500,
            "top_features": [],
            "thresholds": {"warn": 0.1, "fail": 0.2},
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_drift_status()

        assert result is not None
        assert result["status"] == "ok"
        assert result["reference_size"] == 1000

    @patch("ui.data_service.requests.get")
    def test_timeout_returns_none(self, mock_get):
        """Timeout should return None."""
        import requests

        mock_get.side_effect = requests.Timeout("Request timed out")

        result = fetch_drift_status()

        assert result is None

    @patch("ui.data_service.requests.get")
    def test_error_returns_none(self, mock_get):
        """HTTP error should return None."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Not Found"
        )
        mock_get.return_value = mock_response

        result = fetch_drift_status()

        assert result is None

    @patch("ui.data_service.requests.get")
    def test_connection_error_returns_none(self, mock_get):
        """Connection error should return None."""
        import requests

        mock_get.side_effect = requests.ConnectionError("Connection failed")

        result = fetch_drift_status()

        assert result is None


class TestRenderDriftPanel:
    """Tests for render_drift_panel function."""

    @patch("ui.app.fetch_drift_status")
    @patch("ui.app.st")
    def test_renders_success_status(self, mock_st, mock_fetch):
        """OK status should render success message."""
        from ui.app import render_drift_panel

        # Mock st.columns to return 3 column mocks
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]

        mock_fetch.return_value = {
            "status": "ok",
            "computed_at": "2024-01-01T12:00:00+00:00",
            "cached": False,
            "reference_window": "Production model",
            "current_window": "Last 24 hours",
            "reference_size": 1000,
            "live_size": 500,
            "top_features": [],
            "thresholds": {"warn": 0.1, "fail": 0.2},
        }

        render_drift_panel()

        # Should call st.success for ok status
        success_calls = [
            call for call in mock_st.method_calls if call[0] == "success"
        ]
        assert len(success_calls) > 0

    @patch("ui.app.fetch_drift_status")
    @patch("ui.app.st")
    def test_renders_warning_status(self, mock_st, mock_fetch):
        """Warn status should render warning message."""
        from ui.app import render_drift_panel

        # Mock st.columns to return 3 column mocks
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]

        mock_fetch.return_value = {
            "status": "warn",
            "computed_at": "2024-01-01T12:00:00+00:00",
            "cached": False,
            "reference_window": "Production model",
            "current_window": "Last 24 hours",
            "reference_size": 1000,
            "live_size": 500,
            "top_features": [],
            "thresholds": {"warn": 0.1, "fail": 0.2},
        }

        render_drift_panel()

        # Should call st.warning for warn status
        warning_calls = [
            call for call in mock_st.method_calls if call[0] == "warning"
        ]
        assert len(warning_calls) > 0

    @patch("ui.app.fetch_drift_status")
    @patch("ui.app.st")
    def test_renders_error_status(self, mock_st, mock_fetch):
        """Fail status should render error message."""
        from ui.app import render_drift_panel

        # Mock st.columns to return 3 column mocks
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]

        mock_fetch.return_value = {
            "status": "fail",
            "computed_at": "2024-01-01T12:00:00+00:00",
            "cached": False,
            "reference_window": "Production model",
            "current_window": "Last 24 hours",
            "reference_size": 1000,
            "live_size": 500,
            "top_features": [],
            "thresholds": {"warn": 0.1, "fail": 0.2},
        }

        render_drift_panel()

        # Should call st.error for fail status
        error_calls = [
            call for call in mock_st.method_calls if call[0] == "error"
        ]
        assert len(error_calls) > 0

    @patch("ui.app.fetch_drift_status")
    @patch("ui.app.st")
    def test_handles_api_unavailable(self, mock_st, mock_fetch):
        """API unavailable should show info message."""
        from ui.app import render_drift_panel

        mock_fetch.return_value = None

        render_drift_panel()

        # Should call st.info when API unavailable
        info_calls = [
            call for call in mock_st.method_calls if call[0] == "info"
        ]
        assert len(info_calls) > 0

    @patch("ui.app.fetch_drift_status")
    @patch("ui.app.st")
    def test_handles_error_in_response(self, mock_st, mock_fetch):
        """Error in response should show warning."""
        from ui.app import render_drift_panel

        mock_fetch.return_value = {
            "status": "unknown",
            "error": "No reference data available",
        }

        render_drift_panel()

        # Should call st.warning when error present
        warning_calls = [
            call for call in mock_st.method_calls if call[0] == "warning"
        ]
        assert len(warning_calls) > 0

    @patch("ui.app.fetch_drift_status")
    @patch("ui.app.st")
    @patch("ui.app.pd.DataFrame")
    def test_displays_top_features_table(self, mock_df, mock_st, mock_fetch):
        """Top features should be displayed in a table."""
        from ui.app import render_drift_panel

        # Mock st.columns to return 3 column mocks
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]

        mock_fetch.return_value = {
            "status": "ok",
            "computed_at": "2024-01-01T12:00:00+00:00",
            "cached": False,
            "reference_window": "Production model",
            "current_window": "Last 24 hours",
            "reference_size": 1000,
            "live_size": 500,
            "top_features": [
                {"feature": "velocity_24h", "psi": 0.15, "status": "WARNING"},
                {"feature": "amount_to_avg_ratio_30d", "psi": 0.05, "status": "OK"},
            ],
            "thresholds": {"warn": 0.1, "fail": 0.2},
        }

        render_drift_panel()

        # Should create DataFrame and call st.dataframe
        mock_df.assert_called_once()
        dataframe_calls = [
            call for call in mock_st.method_calls if call[0] == "dataframe"
        ]
        assert len(dataframe_calls) > 0
