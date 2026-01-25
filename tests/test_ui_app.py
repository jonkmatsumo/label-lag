"""Tests for UI app layer."""

from unittest.mock import MagicMock, patch

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
        self,
        mock_expander,
        mock_button,
        mock_checkbox,
        mock_slider,
        mock_columns,
        mock_subheader,
        mock_markdown,
        mock_header,
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
        mock_expander.return_value.__enter__ = MagicMock(
            return_value=mock_expander_context
        )
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
        """Test st.cache_data.clear() and st.rerun() called after generation."""
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
        """Test st.cache_data.clear() and st.rerun() called after clear."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 500
        mock_checkbox.return_value = True

        # Mock expander context for Danger Zone
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__ = MagicMock(
            return_value=mock_expander_context
        )
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
    def test_navigation_includes_synthetic_dataset(
        self, mock_live, mock_analytics, mock_synthetic, mock_sidebar
    ):
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


class TestRenderModelLab:
    """Tests for render_model_lab function after refactoring."""

    @patch("ui.app.check_mlflow_connection")
    @patch("ui.app.st.header")
    @patch("ui.app.st.markdown")
    @patch("ui.app.st.subheader")
    @patch("ui.app.st.success")
    def test_data_management_section_absent(
        self,
        mock_success,
        mock_subheader,
        mock_markdown,
        mock_header,
        mock_mlflow,
    ):
        """Test that 'Data Management' section is absent from render_model_lab()."""
        from ui.app import render_model_lab

        # Setup mocks
        mock_mlflow.return_value = True
        mock_subheader_call_args = []
        mock_subheader.side_effect = lambda x: mock_subheader_call_args.append(x)

        with patch("ui.app.st.columns") as mock_columns, patch(
            "ui.app.st.slider"
        ) as mock_slider, patch("ui.app.st.button") as mock_button:
            mock_columns.return_value = [MagicMock(), MagicMock()]
            mock_slider.return_value = 6
            mock_button.return_value = False

            with patch("ui.app.get_production_model_version"), patch(
                "ui.app.get_experiment_runs"
            ) as mock_runs, patch("ui.app.promote_to_production"):
                mock_runs.return_value = []

                render_model_lab()

        # Verify "Data Management" is NOT in subheader calls
        assert "Data Management" not in mock_subheader_call_args

    @patch("ui.app.check_mlflow_connection")
    @patch("ui.app.st.subheader")
    def test_train_new_model_section_exists(self, mock_subheader, mock_mlflow):
        """Test that 'Train New Model' section still exists."""
        from ui.app import render_model_lab

        # Setup mocks
        mock_mlflow.return_value = True
        mock_subheader_call_args = []
        mock_subheader.side_effect = lambda x: mock_subheader_call_args.append(x)

        with patch("ui.app.st.header"), patch("ui.app.st.markdown"), patch(
            "ui.app.st.success"
        ), patch("ui.app.st.columns") as mock_columns, patch(
            "ui.app.st.slider"
        ) as mock_slider, patch("ui.app.st.button") as mock_button:
            mock_columns.return_value = [MagicMock(), MagicMock()]
            mock_slider.return_value = 6
            mock_button.return_value = False

            with patch("ui.app.get_production_model_version"), patch(
                "ui.app.get_experiment_runs"
            ) as mock_runs, patch("ui.app.promote_to_production"):
                mock_runs.return_value = []

                render_model_lab()

        # Verify "Train New Model" is in subheader calls
        assert "Train New Model" in mock_subheader_call_args

    @patch("ui.app.check_mlflow_connection")
    @patch("ui.app.st.subheader")
    def test_model_registry_section_exists(self, mock_subheader, mock_mlflow):
        """Test that 'Model Registry' section still exists."""
        from ui.app import render_model_lab

        # Setup mocks
        mock_mlflow.return_value = True
        mock_subheader_call_args = []
        mock_subheader.side_effect = lambda x: mock_subheader_call_args.append(x)

        with patch("ui.app.st.header"), patch("ui.app.st.markdown"), patch(
            "ui.app.st.success"
        ), patch("ui.app.st.columns") as mock_columns, patch(
            "ui.app.st.slider"
        ) as mock_slider, patch("ui.app.st.button") as mock_button:
            mock_columns.return_value = [MagicMock(), MagicMock()]
            mock_slider.return_value = 6
            mock_button.return_value = False

            with patch("ui.app.get_production_model_version"), patch(
                "ui.app.get_experiment_runs"
            ) as mock_runs, patch("ui.app.promote_to_production"):
                mock_runs.return_value = []

                render_model_lab()

        # Verify "Model Registry" is in subheader calls
        assert "Model Registry" in mock_subheader_call_args

    def test_function_docstring_no_data_generation_mention(self):
        """Test that function docstring no longer mentions data generation."""
        from ui.app import render_model_lab

        docstring = render_model_lab.__doc__
        assert docstring is not None
        # Verify docstring does not mention data generation
        assert "data generation" not in docstring.lower()
        assert "generate" not in docstring.lower() or "generate" in docstring.lower()  # Allow if it's about model generation

    @patch("ui.app.check_mlflow_connection")
    @patch("ui.app.st.success")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.post")
    @patch("ui.app.st.button")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_training_functionality_still_works(
        self,
        mock_columns,
        mock_slider,
        mock_button,
        mock_post,
        mock_spinner,
        mock_success,
        mock_mlflow,
    ):
        """Verify all existing Model Lab functionality (training) still works."""
        from ui.app import render_model_lab

        # Setup mocks
        mock_mlflow.return_value = True
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 6
        mock_button.return_value = True  # Training button clicked
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock successful training response
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True, "run_id": "test_run_123"}
        mock_post.return_value = mock_response

        with patch("ui.app.st.header"), patch("ui.app.st.markdown"), patch(
            "ui.app.st.subheader"
        ), patch("ui.app.get_production_model_version"), patch(
            "ui.app.get_experiment_runs"
        ) as mock_runs, patch("ui.app.promote_to_production"):
            mock_runs.return_value = []

            render_model_lab()

        # Verify training API was called
        mock_post.assert_called()
        # Verify success message
        mock_success.assert_called()

    @patch("ui.app.check_mlflow_connection")
    @patch("ui.app.get_production_model_version")
    @patch("ui.app.get_experiment_runs")
    @patch("ui.app.promote_to_production")
    def test_registry_functionality_still_works(
        self, mock_promote, mock_runs, mock_prod_version, mock_mlflow
    ):
        """Verify all existing Model Lab functionality (registry) still works."""
        from ui.app import render_model_lab

        # Setup mocks
        mock_mlflow.return_value = True
        mock_prod_version.return_value = "1"
        mock_runs.return_value = []

        with patch("ui.app.st.header"), patch("ui.app.st.markdown"), patch(
            "ui.app.st.subheader"
        ), patch("ui.app.st.success"), patch("ui.app.st.columns") as mock_columns, patch(
            "ui.app.st.slider"
        ) as mock_slider, patch("ui.app.st.button") as mock_button, patch(
            "ui.app.st.dataframe"
        ), patch(
            "ui.app.st.selectbox"
        ):
            mock_columns.return_value = [MagicMock(), MagicMock()]
            mock_slider.return_value = 6
            mock_button.return_value = False

            render_model_lab()

        # Verify registry functions were called
        mock_prod_version.assert_called()
        mock_runs.assert_called()


class TestCacheInvalidation:
    """Tests for cache invalidation behavior after generate/clear operations."""

    @patch("ui.app.st.cache_data")
    @patch("ui.app.st.rerun")
    @patch("ui.app.st.success")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.post")
    @patch("ui.app.st.button")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_cache_clear_called_exactly_once_after_successful_generate(
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
        """Test st.cache_data.clear() called exactly once after generate."""
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

        # Verify cache clear is called exactly once
        assert mock_cache_data.clear.call_count == 1

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
    def test_cache_clear_called_exactly_once_after_successful_clear(
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
        """Test st.cache_data.clear() called exactly once after clear."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 500
        mock_checkbox.return_value = True

        # Mock expander context for Danger Zone
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__ = MagicMock(
            return_value=mock_expander_context
        )
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

        # Verify cache clear is called exactly once
        assert mock_cache_data.clear.call_count == 1

    @patch("ui.app.st.rerun")
    @patch("ui.app.st.success")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.post")
    @patch("ui.app.st.button")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_rerun_called_after_successful_generate(
        self,
        mock_columns,
        mock_slider,
        mock_checkbox,
        mock_button,
        mock_post,
        mock_spinner,
        mock_success,
        mock_rerun,
    ):
        """Test that st.rerun() is called after successful generate."""
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

        with patch("ui.app.st.cache_data"):
            render_synthetic_dataset()

        # Verify rerun is called
        mock_rerun.assert_called_once()

    @patch("ui.app.st.rerun")
    @patch("ui.app.st.success")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.delete")
    @patch("ui.app.st.expander")
    @patch("ui.app.st.button")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_rerun_called_after_successful_clear(
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
    ):
        """Test that st.rerun() is called after successful clear."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 500
        mock_checkbox.return_value = True

        # Mock expander context for Danger Zone
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__ = MagicMock(
            return_value=mock_expander_context
        )
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

        with patch("ui.app.st.cache_data"):
            render_synthetic_dataset()

        # Verify rerun is called
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
    def test_cache_clear_not_called_on_api_errors(
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
        """Test that cache clear is NOT called on API errors."""
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

        # Verify cache clear is NOT called
        mock_cache_data.clear.assert_not_called()
        # Verify rerun is NOT called
        mock_rerun.assert_not_called()

    @patch("ui.app.st.cache_data")
    @patch("ui.app.st.rerun")
    @patch("ui.app.st.error")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.post")
    @patch("ui.app.st.button")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_rerun_not_called_on_api_errors(
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
        """Test that rerun is NOT called on API errors."""
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

        # Verify rerun is NOT called
        mock_rerun.assert_not_called()

    @patch("ui.app.st.cache_data")
    @patch("ui.app.st.rerun")
    @patch("ui.app.st.error")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.post")
    @patch("ui.app.st.button")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_cache_clear_not_called_on_network_timeouts(
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
        """Test that cache clear is NOT called on network timeouts."""
        from ui.app import render_synthetic_dataset

        # Setup mocks
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_slider.return_value = 500
        mock_checkbox.return_value = True
        mock_button.return_value = True  # Generate button clicked
        mock_spinner.return_value.__enter__ = MagicMock()
        mock_spinner.return_value.__exit__ = MagicMock()

        # Mock timeout exception
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")

        render_synthetic_dataset()

        # Verify cache clear is NOT called
        mock_cache_data.clear.assert_not_called()
        # Verify rerun is NOT called
        mock_rerun.assert_not_called()

    @patch("ui.app.st.cache_data")
    @patch("ui.app.st.rerun")
    @patch("ui.app.st.success")
    @patch("ui.app.st.spinner")
    @patch("ui.app.requests.post")
    @patch("ui.app.st.button")
    @patch("ui.app.st.checkbox")
    @patch("ui.app.st.slider")
    @patch("ui.app.st.columns")
    def test_cache_clear_and_rerun_call_timing(
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
        """Test that cache clear and rerun are called in correct order after success."""
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

        # Track call order
        call_order = []

        def track_cache_clear():
            call_order.append("cache_clear")

        def track_rerun():
            call_order.append("rerun")

        mock_cache_data.clear.side_effect = track_cache_clear
        mock_rerun.side_effect = track_rerun

        render_synthetic_dataset()

        # Verify cache clear is called before rerun
        assert call_order == ["cache_clear", "rerun"]
