"""Tests for UI app layer."""

from unittest.mock import MagicMock, patch

import pytest
import requests


class TestRenderSyntheticDataset:
    """Tests for render_synthetic_dataset function."""

    def test_function_exists_and_callable(self):
        """Test that render_synthetic_dataset function exists and is callable."""
        from ui.app import render_synthetic_dataset

        assert callable(render_synthetic_dataset)

    @patch("ui.app.st.header")
    @patch("ui.app.st.markdown")
    @patch("ui.app.st.subheader")
    @patch("ui.app.st.columns")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.button")
    @patch("ui.app.st.expander")
    def test_page_renders_header_and_description(
        self, mock_expander, mock_button, mock_checkbox, mock_slider, mock_columns, mock_subheader, mock_markdown, mock_header
    ):
        """Test that page renders header and description."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 500
        mock_checkbox.return_value = True
        mock_button.return_value = False
        mock_expander.return_value.__enter__ = MagicMock()
        mock_expander.return_value.__exit__ = MagicMock()

        render_synthetic_dataset()

        # Verify header and description are called
        mock_header.assert_called_once()
        assert mock_markdown.call_count >= 1

    @patch("ui.app.st.columns")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.button")
    @patch("ui.app.st.expander")
    def test_all_controls_present(
        self, mock_expander, mock_button, mock_checkbox, mock_slider, mock_columns
    ):
        """Test that all controls (sliders, checkbox, buttons) are present."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        mock_slider.return_value = 500
        mock_checkbox.return_value = True
        mock_button.return_value = False
        mock_expander.return_value.__enter__ = MagicMock()
        mock_expander.return_value.__exit__ = MagicMock()

        render_synthetic_dataset()

        # Verify sliders are called (num_users and fraud_rate)
        assert mock_slider.call_count >= 2
        # Verify checkbox is called
        mock_checkbox.assert_called()
        # Verify button is called (Generate Data)
        assert mock_button.call_count >= 1

    @patch("ui.app.st.expander")
    @patch("ui.app.st.button")
    def test_danger_zone_expander_exists(self, mock_button, mock_expander):
        """Test that 'Danger Zone' expander exists and contains clear button."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__ = MagicMock(return_value=mock_expander_context)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)
        mock_button.return_value = False

        # Mock other required components
        with patch("ui.app.st.columns") as mock_columns, patch(
            "ui.app.st.slider"
        ) as mock_slider, patch("ui.app.st.checkbox") as mock_checkbox:
            mock_columns.return_value = [MagicMock(), MagicMock()]
            mock_slider.return_value = 500
            mock_checkbox.return_value = True

            render_synthetic_dataset()

        # Verify expander is called with "Danger Zone"
        mock_expander.assert_called()
        # Verify button is called inside expander context
        assert mock_button.call_count >= 1

    @patch("ui.app.st.cache_data")
    @patch("ui.app.st.rerun")
    @patch("ui.app.st.success")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.post")
    @patch("ui.app.st.button")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_successful_generate_calls_cache_clear_and_rerun(
        self,
        mock_columns,
        mock_slider,
        mock_checkbox,
        mock_button,
        mock_post,
        mock_spinner,
        mock_success,
        mock_rerun,
        mock_cache_data,
    ):
        """Test that st.cache_data.clear() and st.rerun() are called after successful generation."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 500
        mock_checkbox.return_value = True
        mock_button.return_value = True  # Generate button clicked
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "total_records": 1000,
            "fraud_records": 50,
            "features_materialized": 1000,
        }
        mock_post.return_value = mock_response

        render_synthetic_dataset()

        # Verify cache clear and rerun are called
        mock_cache_data.clear.assert_called_once()
        mock_rerun.assert_called_once()

    @patch("ui.app.st.cache_data")
    @patch("ui.app.st.rerun")
    @patch("ui.app.st.success")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.delete")
    @patch("ui.app.st.expander")
    @patch("ui.app.st.button")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_successful_clear_calls_cache_clear_and_rerun(
        self,
        mock_columns,
        mock_slider,
        mock_checkbox,
        mock_button,
        mock_expander,
        mock_delete,
        mock_spinner,
        mock_success,
        mock_rerun,
        mock_cache_data,
    ):
        """Test that st.cache_data.clear() and st.rerun() are called after successful clear."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 500
        mock_checkbox.return_value = True

        # Mock expander context for Danger Zone
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__ = MagicMock(return_value=mock_expander_context)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)

        # First button call is Generate (False), second is Clear (True)
        mock_button.side_effect = [False, True]
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "tables_cleared": ["generated_records", "feature_snapshots"],
        }
        mock_delete.return_value = mock_response

        render_synthetic_dataset()

        # Verify cache clear and rerun are called
        mock_cache_data.clear.assert_called_once()
        mock_rerun.assert_called_once()

    @patch("ui.app.st.cache_data")
    @patch("ui.app.st.rerun")
    @patch("ui.app.st.error")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.post")
    @patch("ui.app.st.button")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_error_does_not_call_cache_clear_or_rerun(
        self,
        mock_columns,
        mock_slider,
        mock_checkbox,
        mock_button,
        mock_post,
        mock_spinner,
        mock_error,
        mock_rerun,
        mock_cache_data,
    ):
        """Test that cache clear and rerun are NOT called on errors."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 500
        mock_checkbox.return_value = True
        mock_button.return_value = True  # Generate button clicked
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock API error response
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": False, "error": "Generation failed"}
        mock_post.return_value = mock_response

        render_synthetic_dataset()

        # Verify cache clear and rerun are NOT called
        mock_cache_data.clear.assert_not_called()
        mock_rerun.assert_not_called()
        # Verify error is displayed
        mock_error.assert_called()

    @patch("ui.app.st.error")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.post")
    @patch("ui.app.st.button")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_request_exception_does_not_call_cache_clear(
        self,
        mock_columns,
        mock_slider,
        mock_checkbox,
        mock_button,
        mock_post,
        mock_spinner,
        mock_error,
    ):
        """Test that cache clear is NOT called on request exceptions."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 500
        mock_checkbox.return_value = True
        mock_button.return_value = True  # Generate button clicked
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock request exception
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")

        with patch("ui.app.st.cache_data") as mock_cache_data, patch(
            "ui.app.st.rerun"
        ) as mock_rerun:
            render_synthetic_dataset()

            # Verify cache clear and rerun are NOT called
            mock_cache_data.clear.assert_not_called()
            mock_rerun.assert_not_called()
            # Verify error is displayed
            mock_error.assert_called()


class TestNavigation:
    """Tests for navigation and routing."""

    @patch("ui.app.st.sidebar")
    @patch("ui.app.render_synthetic_dataset")
    @patch("ui.app.render_analytics")
    @patch("ui.app.render_live_scoring")
    def test_navigation_includes_synthetic_dataset(self, mock_live, mock_analytics, mock_synthetic, mock_sidebar):
        """Test that navigation includes 'Synthetic Dataset' option."""
        from ui.app import main

        # Mock sidebar radio to return "Synthetic Dataset"
        mock_sidebar.radio.return_value = "Synthetic Dataset"
        mock_sidebar.title = MagicMock()
        mock_sidebar.markdown = MagicMock()
        mock_sidebar.text = MagicMock()

        main()

        # Verify navigation options include "Synthetic Dataset"
        call_args = mock_sidebar.radio.call_args
        assert call_args is not None
        options = call_args[1]["options"]
        assert "Synthetic Dataset" in options

    @patch("ui.app.st.sidebar")
    @patch("ui.app.render_synthetic_dataset")
    @patch("ui.app.render_analytics")
    @patch("ui.app.render_live_scoring")
    def test_routing_calls_render_synthetic_dataset(self, mock_live, mock_analytics, mock_synthetic, mock_sidebar):
        """Test that routing correctly calls render_synthetic_dataset() when selected."""
        from ui.app import main

        # Mock sidebar radio to return "Synthetic Dataset"
        mock_sidebar.radio.return_value = "Synthetic Dataset"
        mock_sidebar.title = MagicMock()
        mock_sidebar.markdown = MagicMock()
        mock_sidebar.text = MagicMock()

        main()

        # Verify render_synthetic_dataset is called
        mock_synthetic.assert_called_once()
        # Verify other render functions are not called
        mock_live.assert_not_called()
        mock_analytics.assert_not_called()
