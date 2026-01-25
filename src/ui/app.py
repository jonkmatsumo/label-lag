"""ACH Risk Inspector Dashboard.

A Streamlit dashboard for fraud risk analysis with four modes:
- Live Scoring: Real-time transaction evaluation via API
- Historical Analytics: Analysis of historical data from database
- Synthetic Dataset: Generate and manage synthetic training data
- Model Lab: Train models and manage the model registry

NOTE: This service is isolated and does NOT import from src.model or src.generator.
"""

import json
import os
import time

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_service import (
    check_api_health,
    fetch_daily_stats,
    fetch_feature_sample,
    fetch_fraud_summary,
    fetch_overview_metrics,
    fetch_recent_alerts,
    fetch_schema_summary,
    fetch_transaction_details,
    predict_risk,
)
from mlflow_utils import (
    check_mlflow_connection,
    get_experiment_runs,
    get_model_versions,
    get_production_model_version,
    promote_to_production,
)
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np

# Configuration from environment
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://synthetic:synthetic_dev_password@localhost:5432/synthetic_data",
)

# Page configuration
st.set_page_config(
    page_title="ACH Risk Inspector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_live_scoring() -> None:
    """Render the Live Scoring (API) page.

    This page allows users to submit transactions for real-time
    fraud risk evaluation via the signal API.
    """
    st.header("Live Scoring")
    st.markdown("Submit transactions for real-time fraud risk evaluation.")

    # Show current model status
    api_health = check_api_health()
    if api_health:
        model_loaded = api_health.get("model_loaded", False)
        model_version = api_health.get("version", "unknown")

        if model_loaded:
            st.success(f"Live Model: **{model_version}** (ML model active)")
        else:
            st.warning(f"Live Model: **{model_version}** (rule-based fallback)")
    else:
        st.error("API unavailable - cannot determine model status")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Input")

        user_id = st.text_input("User ID", value="user_001")
        amount = st.number_input(
            "Amount", min_value=0.01, value=100.00, step=0.01, format="%.2f"
        )
        currency = st.text_input("Currency", value="USD", disabled=True)

        analyze_clicked = st.button("Analyze Risk", type="primary")

    with col2:
        st.subheader("Risk Assessment")

        if analyze_clicked:
            # Measure API latency
            start_time = time.time()
            result = predict_risk(user_id, amount, currency)
            elapsed_ms = (time.time() - start_time) * 1000

            if result is None:
                st.error("API request failed. Is the API server running?")
                st.caption(f"Latency: {elapsed_ms:.0f}ms")
            else:
                score = result.get("score", 0)

                # Score gauge with color-coded risk level
                if score < 10:
                    st.markdown(
                        "<h1 style='color: #2ecc71; text-align: center;'>LOW RISK</h1>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<h2 style='color: #2ecc71; text-align: center;'>"
                        f"Score: {score}</h2>",
                        unsafe_allow_html=True,
                    )
                elif score < 80:
                    st.markdown(
                        "<h1 style='color: #f39c12; text-align: center;'>"
                        "MEDIUM RISK</h1>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<h2 style='color: #f39c12; text-align: center;'>"
                        f"Score: {score}</h2>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<h1 style='color: #e74c3c; text-align: center;'>"
                        "HIGH RISK</h1>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<h2 style='color: #e74c3c; text-align: center;'>"
                        f"Score: {score}</h2>",
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                # Risk components
                risk_components = result.get("risk_components", [])
                if risk_components:
                    st.markdown("**Risk Factors:**")
                    for component in risk_components:
                        label = component.get("label", "unknown")
                        st.caption(f"- {label}")
                else:
                    st.caption("No specific risk factors identified.")

                # Latency display
                st.markdown("---")
                st.caption(f"Latency: {elapsed_ms:.0f}ms")

                # Raw JSON expander
                with st.expander("View Raw API Response"):
                    st.json(json.dumps(result, indent=2, default=str))
        else:
            st.markdown("*Submit a transaction to see results*")


def _render_model_selector() -> None:
    """Render model version selector with current production indicator.

    Shows a dropdown of all available model versions with the production
    model clearly marked. Currently for display purposes - future versions
    could use this to compare model performance.
    """
    # Get API health for live model info
    api_health = check_api_health()
    live_model = None
    if api_health and api_health.get("model_loaded"):
        live_model = api_health.get("version", "unknown")

    # Get all model versions
    all_versions = get_model_versions()

    if not all_versions:
        st.info("No models registered yet. Train a model in Model Lab to get started.")
        return

    # Build options list with indicators
    options = []
    for v in sorted(all_versions, key=lambda x: int(x["version"]), reverse=True):
        version = v["version"]
        stage = v["stage"]

        # Build label with indicators
        label = f"v{version}"
        indicators = []

        if stage == "Production":
            indicators.append("PRODUCTION")
        if live_model and live_model == f"v{version}":
            indicators.append("LIVE")

        if indicators:
            label += f" ({', '.join(indicators)})"
        elif stage != "None":
            label += f" ({stage})"

        options.append({"label": label, "version": version, "stage": stage})

    # Display current live model status
    col1, col2 = st.columns([2, 3])

    with col1:
        if live_model:
            st.success(f"Live Model: **{live_model}**")
        else:
            st.warning("No ML model loaded (using rules)")

    with col2:
        # Model version dropdown
        selected_idx = st.selectbox(
            "View Model Version",
            options=range(len(options)),
            format_func=lambda i: options[i]["label"],
            help="Select a model version to view details. LIVE = serving.",
        )

        if selected_idx is not None:
            selected = options[selected_idx]
            # Store in session state for potential future use
            st.session_state["selected_model_version"] = selected["version"]


def render_analytics() -> None:
    """Render the Historical Analytics (DB) page.

    This page provides analytics and visualizations based on
    historical transaction data stored in the database.
    """
    st.header("Historical Analytics")
    st.markdown("Analyze historical transaction patterns and fraud metrics.")

    # --- Model Selection ---
    _render_model_selector()

    st.markdown("---")

    # Fetch data
    summary = fetch_fraud_summary()
    daily_stats = fetch_daily_stats(days=30)
    transactions = fetch_transaction_details(days=7)
    alerts = fetch_recent_alerts(limit=50)

    # --- Global Metrics ---
    st.subheader("Global Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Transactions Analyzed",
            value=f"{summary['total_transactions']:,}",
        )

    with col2:
        fraud_delta = (
            f"{summary['fraud_rate']:.2f}%" if summary["fraud_rate"] > 0 else None
        )
        st.metric(
            label="Detected Fraud (High Risk)",
            value=f"{summary['total_fraud']:,}",
            delta=fraud_delta,
            delta_color="inverse",
        )

    with col3:
        # Estimate false positive rate based on alerts vs actual fraud
        # This is a rough estimate: alerts that aren't fraud / total alerts
        if len(alerts) > 0 and "is_fraudulent" in alerts.columns:
            true_positives = alerts["is_fraudulent"].sum()
            false_positives = len(alerts) - true_positives
            fpr = (false_positives / len(alerts) * 100) if len(alerts) > 0 else 0
            st.metric(
                label="False Positive Rate (Est)",
                value=f"{fpr:.1f}%",
                help="Percentage of high-risk alerts that are not actual fraud",
            )
        else:
            st.metric(
                label="False Positive Rate (Est)",
                value="--",
                help="No alert data available",
            )

    st.markdown("---")

    # --- Time Series Visualization ---
    st.subheader("Transaction Volume & Fraud Trends")

    if len(daily_stats) > 0:
        # Create dual-axis chart: bars for volume, line for fraud
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Bar chart for transaction volume
        fig.add_trace(
            go.Bar(
                x=daily_stats["date"],
                y=daily_stats["total_transactions"],
                name="Transaction Volume",
                marker_color="#3498db",
                opacity=0.7,
            ),
            secondary_y=False,
        )

        # Line chart for fraud count
        fig.add_trace(
            go.Scatter(
                x=daily_stats["date"],
                y=daily_stats["fraud_count"],
                name="Fraud Count",
                mode="lines+markers",
                line={"color": "#e74c3c", "width": 3},
                marker={"size": 8},
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title="Daily Transaction Volume with Fraud Overlay",
            xaxis_title="Date",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            height=400,
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Transaction Count", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Count", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No daily statistics available. Generate data to see trends.")

    st.markdown("---")

    # --- Amount Distribution ---
    st.subheader("Transaction Amount Distribution")

    if len(transactions) > 0 and "amount" in transactions.columns:
        # Create a copy and add fraud label for visualization
        viz_data = transactions.copy()
        if "is_fraudulent" in viz_data.columns:
            viz_data["Fraud Status"] = viz_data["is_fraudulent"].map(
                {True: "Fraudulent", False: "Legitimate"}
            )
        else:
            viz_data["Fraud Status"] = "Unknown"

        fig = px.histogram(
            viz_data,
            x="amount",
            color="Fraud Status",
            nbins=50,
            title="Transaction Amount Distribution by Fraud Status",
            labels={"amount": "Transaction Amount ($)", "count": "Frequency"},
            color_discrete_map={
                "Fraudulent": "#e74c3c",
                "Legitimate": "#2ecc71",
                "Unknown": "#95a5a6",
            },
            opacity=0.7,
            barmode="overlay",
        )
        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transaction data available. Generate data to see distribution.")

    st.markdown("---")

    # --- Recent Alerts Table ---
    st.subheader("Recent High-Risk Alerts")

    if len(alerts) > 0:
        # Format the display columns
        display_cols = [
            "record_id",
            "user_id",
            "created_at",
            "amount",
            "computed_risk_score",
            "is_fraudulent",
            "fraud_type",
        ]
        available_cols = [c for c in display_cols if c in alerts.columns]

        if available_cols:
            display_df = alerts[available_cols].copy()

            # Rename columns for display
            column_names = {
                "record_id": "Record ID",
                "user_id": "User ID",
                "created_at": "Timestamp",
                "amount": "Amount ($)",
                "computed_risk_score": "Risk Score",
                "is_fraudulent": "Confirmed Fraud",
                "fraud_type": "Fraud Type",
            }
            display_df = display_df.rename(
                columns={k: v for k, v in column_names.items() if k in display_df}
            )

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(f"Showing {len(alerts)} high-risk transactions (score >= 80)")
        else:
            st.warning("Alert data structure unexpected.")
    else:
        st.info(
            "No high-risk alerts found. This could mean no risky transactions "
            "or no data has been generated yet."
        )


def _get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Get list of numeric column names, excluding record_id and boolean columns.

    Args:
        df: DataFrame to analyze.

    Returns:
        List of numeric column names.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Exclude record_id and boolean columns
    exclude = ["record_id"]
    return [col for col in numeric_cols if col not in exclude]


def _get_categorical_columns(df: pd.DataFrame, max_cardinality: int = 50) -> list[str]:
    """Get list of categorical column names with cardinality guardrails.

    Args:
        df: DataFrame to analyze.
        max_cardinality: Maximum unique values to consider a column categorical.

    Returns:
        List of categorical column names (object/string/bool with low cardinality).
    """
    categorical_cols = []
    for col in df.columns:
        if col in ["record_id"]:
            continue
        dtype = df[col].dtype
        if dtype in ["object", "string", "bool"] or dtype.name == "category":
            nunique = df[col].nunique()
            if nunique <= max_cardinality and nunique > 1:
                categorical_cols.append(col)
    return categorical_cols


def render_synthetic_dataset() -> None:
    """Render the Synthetic Dataset page.

    This page provides controls for generating and managing
    synthetic training data.
    """
    st.header("Synthetic Dataset")
    st.markdown("Generate and manage synthetic training data for model development.")

    # --- Dataset Overview section ---
    st.subheader("Dataset Overview")

    try:
        overview = fetch_overview_metrics()

        col1, col2, col3 = st.columns(3)

        with col1:
            total_records = overview.get("total_records", 0)
            fraud_records = overview.get("fraud_records", 0)
            st.metric(
                label="Total Records",
                value=f"{total_records:,}" if total_records > 0 else "—",
            )
            st.metric(
                label="Fraud Records",
                value=f"{fraud_records:,}" if fraud_records > 0 else "—",
            )

        with col2:
            fraud_rate = overview.get("fraud_rate", 0.0)
            unique_users = overview.get("unique_users", 0)
            st.metric(
                label="Fraud Rate",
                value=f"{fraud_rate:.2f}%" if fraud_rate > 0 else "—",
            )
            st.metric(
                label="Unique Users",
                value=f"{unique_users:,}" if unique_users > 0 else "—",
            )

        with col3:
            min_ts = overview.get("min_transaction_timestamp")
            max_ts = overview.get("max_transaction_timestamp")
            if min_ts and max_ts:
                date_range = f"{min_ts.strftime('%Y-%m-%d')} → {max_ts.strftime('%Y-%m-%d')}"
            else:
                date_range = "—"
            st.metric(
                label="Transaction Date Range",
                value=date_range,
            )
    except Exception as e:
        st.error(f"Error loading dataset overview: {e}")

    st.markdown("---")

    # --- Schema Summary section ---
    st.subheader("Schema Summary")

    try:
        schema_df = fetch_schema_summary()

        if schema_df.empty:
            st.info("No schema information available.")
        else:
            # Display only relevant columns
            display_cols = ["table_name", "column_name", "data_type", "is_nullable"]
            available_cols = [col for col in display_cols if col in schema_df.columns]
            if available_cols:
                st.dataframe(
                    schema_df[available_cols],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(schema_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading schema summary: {e}")

    st.markdown("---")

    # --- Generate Dataset section ---
    st.subheader("Generate Dataset")

    col1, col2 = st.columns(2)

    with col1:
        num_users = st.slider(
            "Number of Users",
            min_value=100,
            max_value=5000,
            value=500,
            step=100,
            help="Number of unique users to generate",
        )

    with col2:
        fraud_rate = st.slider(
            "Fraud Rate",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Fraction of users with fraud events",
        )

    drop_existing = st.checkbox(
        "Drop existing data before generating",
        value=True,
        help="If checked, existing data will be deleted before generating new data",
    )

    generate_clicked = st.button("Generate Data", type="primary")

    if generate_clicked:
        with st.spinner(f"Generating data for {num_users} users..."):
            try:
                import requests

                response = requests.post(
                    f"{API_BASE_URL}/data/generate",
                    json={
                        "num_users": num_users,
                        "fraud_rate": fraud_rate,
                        "drop_existing": drop_existing,
                    },
                    timeout=300,
                )
                result = response.json()

                if result.get("success"):
                    total = result.get("total_records")
                    fraud = result.get("fraud_records")
                    features = result.get("features_materialized")
                    st.success(
                        f"Generated {total} records ({fraud} fraud). "
                        f"Materialized {features} feature snapshots."
                    )
                    # Clear caches and refresh UI
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error(f"Generation failed: {result.get('error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")

    st.markdown("---")

    # --- Danger Zone section ---
    with st.expander("Danger Zone"):
        clear_clicked = st.button("Clear All Data", type="secondary")

        if clear_clicked:
            with st.spinner("Clearing all data..."):
                try:
                    import requests

                    response = requests.delete(
                        f"{API_BASE_URL}/data/clear",
                        timeout=60,
                    )
                    result = response.json()

                    if result.get("success"):
                        tables = ", ".join(result.get("tables_cleared", []))
                        st.success(f"Cleared tables: {tables}")
                        # Clear caches and refresh UI
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(f"Clear failed: {result.get('error')}")
                except requests.exceptions.RequestException as e:
                    st.error(f"API request failed: {e}")

    st.markdown("---")

    # --- Diagnostics section ---
    st.subheader("Diagnostics")

    tab1, tab2, tab3 = st.tabs(["Distributions", "Missingness", "Outliers"])

    with tab1:
        # Distributions Tab
        with st.spinner("Loading sampled data..."):
            try:
                sample_df = fetch_feature_sample(sample_size=5000, stratify=True)

                if sample_df.empty:
                    st.info("No data available for distribution analysis.")
                else:
                    numeric_cols = _get_numeric_columns(sample_df)

                    if not numeric_cols:
                        st.warning("No numeric columns available for distribution analysis.")
                    else:
                        selected_col = st.selectbox(
                            "Select Feature Column",
                            options=numeric_cols,
                            help="Choose a numeric feature to visualize",
                        )

                        if selected_col:
                            # Histogram
                            color_col = None
                            if "is_fraudulent" in sample_df.columns:
                                color_col = "is_fraudulent"

                            fig_hist = px.histogram(
                                sample_df,
                                x=selected_col,
                                color=color_col,
                                title=f"Distribution of {selected_col}",
                                labels={selected_col: selected_col, "count": "Frequency"},
                                nbins=50,
                            )
                            fig_hist.update_layout(height=400)
                            st.plotly_chart(fig_hist, use_container_width=True)

                            # Box plot
                            fig_box = px.box(
                                sample_df,
                                y=selected_col,
                                color=color_col,
                                title=f"Box Plot of {selected_col}",
                                labels={selected_col: selected_col},
                            )
                            fig_box.update_layout(height=400)
                            st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading distribution data: {e}")

    with tab2:
        # Missingness Tab
        with st.spinner("Loading sampled data..."):
            try:
                sample_df = fetch_feature_sample(sample_size=5000, stratify=True)

                if sample_df.empty:
                    st.info("No data available for missingness analysis.")
                else:
                    # Compute missingness percentage per column
                    missingness_data = []
                    for col in sample_df.columns:
                        missing_count = sample_df[col].isna().sum()
                        missing_pct = (missing_count / len(sample_df) * 100) if len(sample_df) > 0 else 0
                        missingness_data.append(
                            {"column": col, "missingness_pct": missing_pct}
                        )

                    missingness_df = pd.DataFrame(missingness_data)

                    if missingness_df.empty:
                        st.info("No columns available for missingness analysis.")
                    else:
                        # Create horizontal bar chart
                        fig = px.bar(
                            missingness_df,
                            x="missingness_pct",
                            y="column",
                            orientation="h",
                            title="Missingness by Column (%)",
                            labels={
                                "missingness_pct": "Missingness (%)",
                                "column": "Column Name",
                            },
                        )
                        fig.update_layout(height=max(400, len(missingness_df) * 30))
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading missingness data: {e}")

    with tab3:
        # Outliers Tab
        with st.spinner("Loading sampled data..."):
            try:
                sample_df = fetch_feature_sample(sample_size=5000, stratify=True)

                if sample_df.empty:
                    st.info("No data available for outlier analysis.")
                else:
                    numeric_cols = _get_numeric_columns(sample_df)

                    if not numeric_cols:
                        st.warning("No numeric columns available for outlier analysis.")
                    else:
                        selected_col = st.selectbox(
                            "Select Feature Column",
                            options=numeric_cols,
                            help="Choose a numeric feature to analyze for outliers",
                            key="outlier_column",
                        )

                        if selected_col:
                            # Compute IQR-based outliers
                            col_data = sample_df[selected_col].dropna()

                            if len(col_data) == 0:
                                st.warning(f"No valid data in column {selected_col}.")
                            else:
                                Q1 = col_data.quantile(0.25)
                                Q3 = col_data.quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR

                                outliers = col_data[
                                    (col_data < lower_bound) | (col_data > upper_bound)
                                ]
                                outlier_count = len(outliers)
                                outlier_pct = (
                                    (outlier_count / len(col_data) * 100)
                                    if len(col_data) > 0
                                    else 0
                                )

                                # Box plot
                                color_col = None
                                if "is_fraudulent" in sample_df.columns:
                                    color_col = "is_fraudulent"

                                fig = px.box(
                                    sample_df,
                                    y=selected_col,
                                    color=color_col,
                                    title=f"Outlier Detection for {selected_col}",
                                    labels={selected_col: selected_col},
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)

                                # Text summary
                                st.markdown("**Outlier Summary**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Outlier Count", f"{outlier_count:,}")
                                with col2:
                                    st.metric("Outlier Percentage", f"{outlier_pct:.2f}%")
                                with col3:
                                    st.metric(
                                        "Bounds",
                                        f"[{lower_bound:.2f}, {upper_bound:.2f}]",
                                    )
            except Exception as e:
                st.error(f"Error loading outlier data: {e}")

    st.markdown("---")

    # --- Relationships section ---
    st.subheader("Relationships")

    # Settings area
    with st.expander("Settings", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            sample_size = st.number_input(
                "Sample Size",
                min_value=100,
                max_value=50000,
                value=5000,
                step=500,
                help="Number of rows to sample for relationship analysis",
            )

        with col2:
            max_numeric_cols = st.number_input(
                "Max Numeric Columns",
                min_value=5,
                max_value=50,
                value=30,
                step=5,
                help="Maximum number of numeric columns to include in correlation matrices",
            )

        with col3:
            max_categorical_cols = st.number_input(
                "Max Categorical Columns",
                min_value=5,
                max_value=50,
                value=30,
                step=5,
                help="Maximum number of categorical columns for association analysis",
            )

        col4, col5 = st.columns(2)

        with col4:
            categorical_cardinality_threshold = st.number_input(
                "Categorical Cardinality Threshold",
                min_value=2,
                max_value=200,
                value=50,
                step=5,
                help="Maximum unique values to consider a column categorical",
            )

        with col5:
            compute_p_values = st.checkbox(
                "Compute p-values",
                value=False,
                help="Compute statistical significance (slower, requires scipy)",
            )

    # Load sampled data and identify columns
    with st.spinner("Loading sampled data..."):
        try:
            sample_df = fetch_feature_sample(sample_size=sample_size, stratify=True)

            if sample_df.empty:
                st.info("No data available for relationship analysis.")
                sample_df = None
                numeric_cols = []
                categorical_cols = []
                actual_sample_size = 0
            else:
                actual_sample_size = len(sample_df)
                numeric_cols = _get_numeric_columns(sample_df)
                categorical_cols = _get_categorical_columns(
                    sample_df, max_cardinality=categorical_cardinality_threshold
                )

                # Apply column caps
                if len(numeric_cols) > max_numeric_cols:
                    # Select top columns by variance
                    variances = sample_df[numeric_cols].var().sort_values(ascending=False)
                    numeric_cols = variances.head(max_numeric_cols).index.tolist()
                    st.warning(
                        f"Limited to top {max_numeric_cols} numeric columns by variance. "
                        f"({len(_get_numeric_columns(sample_df))} total available)"
                    )

                if len(categorical_cols) > max_categorical_cols:
                    # Select top columns by frequency (most common categories)
                    categorical_cols = categorical_cols[:max_categorical_cols]
                    st.warning(
                        f"Limited to {max_categorical_cols} categorical columns. "
                        f"({len(_get_categorical_columns(sample_df, categorical_cardinality_threshold))} total available)"
                    )

                st.caption(f"Using {actual_sample_size:,} rows for analysis")

        except Exception as e:
            st.error(f"Error loading sampled data: {e}")
            sample_df = None
            numeric_cols = []
            categorical_cols = []
            actual_sample_size = 0

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Numeric ↔ Numeric",
            "Categorical ↔ Categorical",
            "Categorical ↔ Numeric",
            "Target Relations",
            "Top Relationships",
        ]
    )

    # Tab implementations will be added in subsequent commits
    with tab1:
        st.info("Numeric correlation analysis will be implemented in the next commit.")

    with tab2:
        st.info("Categorical association analysis will be implemented in the next commit.")

    with tab3:
        st.info("Categorical-Numeric association analysis will be implemented in the next commit.")

    with tab4:
        st.info("Target relations analysis will be implemented in the next commit.")

    with tab5:
        st.info("Top relationships table will be implemented in the next commit.")


def render_model_lab() -> None:
    """Render the Model Lab page.

    This page provides model training and registry management:
    - Train new models with configurable hyperparameters
    - View experiment runs and metrics
    - Promote models to production
    """
    st.header("Model Lab")
    st.markdown("Train models and manage the model registry.")

    # Check MLflow connection
    mlflow_connected = check_mlflow_connection()
    if not mlflow_connected:
        st.error(
            "Cannot connect to MLflow tracking server. "
            "Make sure the MLflow service is running."
        )
        return

    st.success("Connected to MLflow tracking server")

    # --- Section A: Train New Model ---
    st.subheader("Train New Model")

    col1, col2 = st.columns(2)

    with col1:
        max_depth = st.slider(
            "Max Depth",
            min_value=2,
            max_value=12,
            value=6,
            step=1,
            help="Maximum depth of XGBoost trees",
        )

    with col2:
        training_window = st.slider(
            "Training Window (days)",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="Number of days before today for training cutoff",
        )

    train_clicked = st.button("Start Training", type="primary")

    if train_clicked:
        with st.spinner("Training model... This may take a moment."):
            try:
                import requests

                response = requests.post(
                    f"{API_BASE_URL}/train",
                    json={
                        "max_depth": max_depth,
                        "training_window_days": training_window,
                    },
                    timeout=300,  # Training can take a while
                )
                result = response.json()

                if result.get("success"):
                    st.success(f"Training complete! Run ID: `{result.get('run_id')}`")
                    st.balloons()
                else:
                    st.error(f"Training failed: {result.get('error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")

    st.markdown("---")

    # --- Section C: Model Registry ---
    st.subheader("Model Registry")

    # Show current production model
    prod_version = get_production_model_version()
    if prod_version:
        st.info(f"Current Production Model: Version {prod_version}")
    else:
        st.warning("No production model deployed yet.")

    # Fetch and display experiment runs
    runs_df = get_experiment_runs()

    if len(runs_df) > 0:
        st.markdown("**Experiment Runs** (sorted by PR-AUC)")

        st.dataframe(
            runs_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        # Promote to production
        st.markdown("**Promote to Production**")

        run_ids = runs_df["Run ID"].tolist()
        selected_run = st.selectbox(
            "Select Run ID to promote",
            options=run_ids,
            index=0,
            help="Choose a model run to promote to production",
        )

        promote_clicked = st.button("Promote to Production", type="secondary")

        if promote_clicked and selected_run:
            with st.spinner("Promoting model..."):
                result = promote_to_production(selected_run)

            if result["success"]:
                st.success(result["message"])
            else:
                st.error(result["message"])
    else:
        st.info("No experiment runs found. Train a model to see results here.")


def main() -> None:
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("🔍 ACH Risk Inspector")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "Live Scoring (API)",
            "Historical Analytics (DB)",
            "Synthetic Dataset",
            "Model Lab",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Configuration")
    st.sidebar.text(f"API: {API_BASE_URL}")
    db_display = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else "configured"
    st.sidebar.text(f"DB: {db_display}")

    # Render selected page
    if page == "Live Scoring (API)":
        render_live_scoring()
    elif page == "Historical Analytics (DB)":
        render_analytics()
    elif page == "Synthetic Dataset":
        render_synthetic_dataset()
    else:
        render_model_lab()


if __name__ == "__main__":
    main()
