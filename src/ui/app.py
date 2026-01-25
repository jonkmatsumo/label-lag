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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.data_service import (
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
from ui.mlflow_utils import (
    check_mlflow_connection,
    fetch_artifact_path,
    get_experiment_runs,
    get_model_versions,
    get_production_model_version,
    get_run_artifacts,
    get_run_details,
    promote_to_production,
)

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


def _compute_cramers_v(
    col1: pd.Series, col2: pd.Series, compute_p_value: bool = False
) -> tuple[float, float | None]:
    """Compute Cramér's V association between two categorical columns.

    Args:
        col1: First categorical column.
        col2: Second categorical column.
        compute_p_value: Whether to compute p-value (requires scipy).

    Returns:
        Tuple of (cramers_v, p_value). p_value is None if not computed.
    """
    # Build contingency table
    contingency = pd.crosstab(col1, col2)

    # Compute chi-square statistic manually
    chi2 = 0.0
    n = contingency.sum().sum()
    if n == 0:
        return 0.0, None

    col_sums = contingency.sum(axis=0)
    row_sums = contingency.sum(axis=1)

    for i in range(len(contingency.index)):
        for j in range(len(contingency.columns)):
            observed = contingency.iloc[i, j]
            expected = (row_sums.iloc[i] * col_sums.iloc[j]) / n
            if expected > 0:
                chi2 += ((observed - expected) ** 2) / expected

    # Cramér's V
    min_dim = min(len(contingency.index) - 1, len(contingency.columns) - 1)
    if min_dim > 0 and n > 0:
        cramers_v = np.sqrt(chi2 / (n * min_dim))
    else:
        cramers_v = 0.0

    # P-value (optional, requires scipy)
    p_value = None
    if compute_p_value:
        try:
            from scipy.stats import chi2_contingency

            _, p_value, _, _ = chi2_contingency(contingency)
        except ImportError:
            pass

    return cramers_v, p_value


def _compute_correlation_ratio(categorical: pd.Series, numeric: pd.Series) -> float:
    """Compute correlation ratio (η) for categorical→numeric association.

    Args:
        categorical: Categorical column.
        numeric: Numeric column.

    Returns:
        Correlation ratio η (0 to 1).
    """
    # Remove NaN pairs
    df = pd.DataFrame({"cat": categorical, "num": numeric}).dropna()
    if len(df) == 0:
        return 0.0

    # Group means
    group_means = df.groupby("cat")["num"].mean()
    overall_mean = df["num"].mean()

    # SS_between and SS_total
    ss_between = ((group_means - overall_mean) ** 2 * df.groupby("cat").size()).sum()
    ss_total = ((df["num"] - overall_mean) ** 2).sum()

    if ss_total > 0:
        eta = np.sqrt(ss_between / ss_total)
    else:
        eta = 0.0

    return eta


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
                date_range = (
                    f"{min_ts.strftime('%Y-%m-%d')} → {max_ts.strftime('%Y-%m-%d')}"
                )
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
                        st.warning(
                            "No numeric columns available for distribution analysis."
                        )
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
                                labels={
                                    selected_col: selected_col,
                                    "count": "Frequency",
                                },
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
                        missing_pct = (
                            (missing_count / len(sample_df) * 100)
                            if len(sample_df) > 0
                            else 0
                        )
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
                                q1 = col_data.quantile(0.25)
                                q3 = col_data.quantile(0.75)
                                iqr = q3 - q1
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr

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
                                    st.metric(
                                        "Outlier Percentage", f"{outlier_pct:.2f}%"
                                    )
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
                help=(
                    "Maximum number of numeric columns to include "
                    "in correlation matrices"
                ),
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
                    variances = (
                        sample_df[numeric_cols].var().sort_values(ascending=False)
                    )
                    numeric_cols = variances.head(max_numeric_cols).index.tolist()
                    total_numeric = len(_get_numeric_columns(sample_df))
                    st.warning(
                        f"Limited to top {max_numeric_cols} numeric columns "
                        f"by variance. ({total_numeric} total available)"
                    )

                if len(categorical_cols) > max_categorical_cols:
                    # Select top columns by frequency (most common categories)
                    categorical_cols = categorical_cols[:max_categorical_cols]
                    total_categorical = len(
                        _get_categorical_columns(
                            sample_df, categorical_cardinality_threshold
                        )
                    )
                    st.warning(
                        f"Limited to {max_categorical_cols} categorical columns. "
                        f"({total_categorical} total available)"
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

    # Tab 1: Numeric ↔ Numeric
    with tab1:
        if sample_df is None or len(numeric_cols) < 2:
            if sample_df is None:
                st.info("No data available for numeric correlation analysis.")
            else:
                st.warning(
                    f"Need at least 2 numeric columns for correlation analysis. "
                    f"Found {len(numeric_cols)} numeric column(s)."
                )
        else:
            with st.spinner("Computing correlation matrices..."):
                try:
                    # Compute correlation matrices
                    numeric_df = sample_df[numeric_cols].select_dtypes(
                        include=["int64", "float64"]
                    )

                    pearson_corr = numeric_df.corr(method="pearson")
                    spearman_corr = numeric_df.corr(method="spearman")

                    # Display Pearson heatmap
                    st.markdown("**Pearson Correlation Matrix**")
                    fig_pearson = px.imshow(
                        pearson_corr,
                        labels=dict(x="Column", y="Column", color="Correlation"),
                        x=pearson_corr.columns,
                        y=pearson_corr.columns,
                        color_continuous_scale="RdYlBu_r",
                        aspect="auto",
                        title="Pearson Correlation Matrix",
                    )
                    fig_pearson.update_layout(height=600)
                    st.plotly_chart(fig_pearson, use_container_width=True)

                    # Display Spearman heatmap
                    st.markdown("**Spearman Correlation Matrix**")
                    fig_spearman = px.imshow(
                        spearman_corr,
                        labels=dict(x="Column", y="Column", color="Correlation"),
                        x=spearman_corr.columns,
                        y=spearman_corr.columns,
                        color_continuous_scale="RdYlBu_r",
                        aspect="auto",
                        title="Spearman Correlation Matrix",
                    )
                    fig_spearman.update_layout(height=600)
                    st.plotly_chart(fig_spearman, use_container_width=True)

                    # Top pairs table
                    st.markdown("**Top Correlations**")
                    # Flatten correlation matrices (exclude diagonal)
                    top_pairs = []
                    n_cols = len(numeric_cols)
                    for i in range(n_cols):
                        for j in range(i + 1, n_cols):
                            col_a = numeric_cols[i]
                            col_b = numeric_cols[j]
                            pearson_val = pearson_corr.iloc[i, j]
                            spearman_val = spearman_corr.iloc[i, j]

                            # Skip NaN values
                            if pd.notna(pearson_val) and pd.notna(spearman_val):
                                top_pairs.append(
                                    {
                                        "Column A": col_a,
                                        "Column B": col_b,
                                        "Pearson": round(pearson_val, 4),
                                        "Spearman": round(spearman_val, 4),
                                        "Abs Pearson": abs(pearson_val),
                                    }
                                )

                    if top_pairs:
                        top_pairs_df = pd.DataFrame(top_pairs)
                        top_pairs_df = top_pairs_df.sort_values(
                            "Abs Pearson", ascending=False
                        ).head(20)
                        # Remove the sorting column for display
                        display_df = top_pairs_df.drop(columns=["Abs Pearson"])
                        st.dataframe(
                            display_df, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("No valid correlations found (all NaN).")

                except Exception as e:
                    st.error(f"Error computing correlations: {e}")

    # Tab 2: Categorical ↔ Categorical
    with tab2:
        if sample_df is None or len(categorical_cols) < 2:
            if sample_df is None:
                st.info("No data available for categorical association analysis.")
            else:
                st.warning(
                    f"Need at least 2 categorical columns for association analysis. "
                    f"Found {len(categorical_cols)} categorical column(s)."
                )
        else:
            with st.spinner("Computing Cramér's V associations..."):
                try:
                    associations = []
                    n_cats = len(categorical_cols)

                    for i in range(n_cats):
                        for j in range(i + 1, n_cats):
                            col_a = categorical_cols[i]
                            col_b = categorical_cols[j]

                            # Check cardinality
                            card_a = sample_df[col_a].nunique()
                            card_b = sample_df[col_b].nunique()
                            if (
                                card_a > categorical_cardinality_threshold
                                or card_b > categorical_cardinality_threshold
                            ):
                                continue

                            cramers_v, p_value = _compute_cramers_v(
                                sample_df[col_a],
                                sample_df[col_b],
                                compute_p_value=compute_p_values,
                            )

                            associations.append(
                                {
                                    "Column A": col_a,
                                    "Column B": col_b,
                                    "Cramér's V": round(cramers_v, 4),
                                    "p-value": (
                                        round(p_value, 4)
                                        if p_value is not None
                                        else None
                                    ),
                                }
                            )

                    if associations:
                        assoc_df = pd.DataFrame(associations)
                        assoc_df = assoc_df.sort_values(
                            "Cramér's V", ascending=False
                        ).head(30)
                        st.dataframe(
                            assoc_df, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info(
                            "No valid associations found. This may be due to "
                            "high cardinality or insufficient data."
                        )

                except Exception as e:
                    st.error(f"Error computing categorical associations: {e}")

    # Tab 3: Categorical ↔ Numeric
    with tab3:
        if sample_df is None:
            st.info("No data available for categorical-numeric association analysis.")
        elif len(categorical_cols) == 0:
            st.warning("No categorical columns available for association analysis.")
        elif len(numeric_cols) == 0:
            st.warning("No numeric columns available for association analysis.")
        else:
            with st.spinner("Computing correlation ratios..."):
                try:
                    associations = []

                    for cat_col in categorical_cols:
                        # Check cardinality
                        card = sample_df[cat_col].nunique()
                        if card > categorical_cardinality_threshold:
                            continue

                        for num_col in numeric_cols:
                            eta = _compute_correlation_ratio(
                                sample_df[cat_col], sample_df[num_col]
                            )

                            associations.append(
                                {
                                    "Categorical Column": cat_col,
                                    "Numeric Column": num_col,
                                    "Correlation Ratio (η)": round(eta, 4),
                                }
                            )

                    if associations:
                        assoc_df = pd.DataFrame(associations)
                        assoc_df = assoc_df.sort_values(
                            "Correlation Ratio (η)", ascending=False
                        ).head(30)
                        st.dataframe(
                            assoc_df, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("No valid associations found.")

                except Exception as e:
                    st.error(f"Error computing categorical-numeric associations: {e}")

    # Tab 4: Target Relations
    with tab4:
        if sample_df is None:
            st.info("No data available for target relations analysis.")
        elif "is_fraudulent" not in sample_df.columns:
            st.warning("Target column 'is_fraudulent' not found in sampled data.")
        else:
            with st.spinner("Computing target associations..."):
                try:
                    target_relations = []

                    # Convert target to numeric if boolean
                    target = sample_df["is_fraudulent"]
                    if target.dtype == "bool":
                        target_numeric = target.astype(int)
                    else:
                        target_numeric = target

                    # Numeric features
                    for num_col in numeric_cols:
                        pearson = target_numeric.corr(
                            sample_df[num_col], method="pearson"
                        )
                        spearman = target_numeric.corr(
                            sample_df[num_col], method="spearman"
                        )

                        if pd.notna(pearson) and pd.notna(spearman):
                            target_relations.append(
                                {
                                    "Feature": num_col,
                                    "Type": "Numeric",
                                    "Pearson": round(pearson, 4),
                                    "Spearman": round(spearman, 4),
                                    "Max Abs Corr": max(abs(pearson), abs(spearman)),
                                    "Metric": (
                                        "pearson"
                                        if abs(pearson) >= abs(spearman)
                                        else "spearman"
                                    ),
                                }
                            )

                    # Categorical features
                    for cat_col in categorical_cols:
                        card = sample_df[cat_col].nunique()
                        if card > categorical_cardinality_threshold:
                            continue

                        eta = _compute_correlation_ratio(
                            sample_df[cat_col], target_numeric
                        )

                        if pd.notna(eta):
                            target_relations.append(
                                {
                                    "Feature": cat_col,
                                    "Type": "Categorical",
                                    "Correlation Ratio (η)": round(eta, 4),
                                    "Max Abs Corr": eta,
                                    "Metric": "eta",
                                }
                            )

                    if target_relations:
                        # Check for potential leakage
                        leakage_keywords = [
                            "fraud",
                            "label",
                            "target",
                            "confirmed",
                            "evaluation",
                        ]
                        leakage_threshold = 0.8

                        for rel in target_relations:
                            feature_name = rel["Feature"].lower()
                            max_assoc = rel.get("Max Abs Corr", 0.0)

                            # Check suspicious names
                            name_leakage = any(
                                keyword in feature_name for keyword in leakage_keywords
                            )

                            # Check high association
                            high_assoc = max_assoc > leakage_threshold

                            rel["Potential Leakage"] = name_leakage or high_assoc

                        target_df = pd.DataFrame(target_relations)
                        target_df = target_df.sort_values(
                            "Max Abs Corr", ascending=False
                        )

                        # Display warning if leakage detected
                        leakage_count = target_df["Potential Leakage"].sum()
                        if leakage_count > 0:
                            st.warning(
                                f"⚠️ Potential data leakage detected in "
                                f"{leakage_count} feature(s). "
                                f"High association (>{leakage_threshold}) or "
                                f"suspicious naming patterns detected. "
                                f"Note: This is a heuristic check on synthetic data."
                            )

                        # Format display
                        display_cols = [
                            "Feature",
                            "Type",
                            "Pearson",
                            "Spearman",
                            "Correlation Ratio (η)",
                            "Potential Leakage",
                        ]
                        available_cols = [
                            c for c in display_cols if c in target_df.columns
                        ]
                        display_df = target_df[available_cols].copy()

                        # Highlight leakage rows
                        st.dataframe(
                            display_df, use_container_width=True, hide_index=True
                        )

                        st.caption(
                            "Note: Leakage detection is heuristic. "
                            "Synthetic data may have different characteristics "
                            "than production data."
                        )
                    else:
                        st.info("No valid target associations found.")

                except Exception as e:
                    st.error(f"Error computing target relations: {e}")

    # Tab 5: Top Relationships
    with tab5:
        if sample_df is None:
            st.info("No data available for top relationships analysis.")
        else:
            with st.spinner("Compiling top relationships..."):
                try:
                    top_relationships = []

                    # Collect from Numeric ↔ Numeric
                    if len(numeric_cols) >= 2:
                        try:
                            numeric_df = sample_df[numeric_cols].select_dtypes(
                                include=["int64", "float64"]
                            )
                            pearson_corr = numeric_df.corr(method="pearson")
                            spearman_corr = numeric_df.corr(method="spearman")

                            n_cols = len(numeric_cols)
                            for i in range(n_cols):
                                for j in range(i + 1, n_cols):
                                    col_a = numeric_cols[i]
                                    col_b = numeric_cols[j]
                                    pearson_val = pearson_corr.iloc[i, j]
                                    spearman_val = spearman_corr.iloc[i, j]

                                    if pd.notna(pearson_val):
                                        top_relationships.append(
                                            {
                                                "relationship_type": "nn",
                                                "col_a": col_a,
                                                "col_b": col_b,
                                                "effect_size": abs(pearson_val),
                                                "metric_name": "pearson",
                                                "p_value": None,
                                                "sample_size": actual_sample_size,
                                            }
                                        )
                                    if pd.notna(spearman_val):
                                        top_relationships.append(
                                            {
                                                "relationship_type": "nn",
                                                "col_a": col_a,
                                                "col_b": col_b,
                                                "effect_size": abs(spearman_val),
                                                "metric_name": "spearman",
                                                "p_value": None,
                                                "sample_size": actual_sample_size,
                                            }
                                        )
                        except Exception:
                            pass

                    # Collect from Categorical ↔ Categorical
                    if len(categorical_cols) >= 2:
                        try:
                            n_cats = len(categorical_cols)
                            for i in range(n_cats):
                                for j in range(i + 1, n_cats):
                                    col_a = categorical_cols[i]
                                    col_b = categorical_cols[j]

                                    if (
                                        sample_df[col_a].nunique()
                                        > categorical_cardinality_threshold
                                        or sample_df[col_b].nunique()
                                        > categorical_cardinality_threshold
                                    ):
                                        continue

                                    cramers_v, p_value = _compute_cramers_v(
                                        sample_df[col_a],
                                        sample_df[col_b],
                                        compute_p_value=compute_p_values,
                                    )

                                    if pd.notna(cramers_v):
                                        top_relationships.append(
                                            {
                                                "relationship_type": "cc",
                                                "col_a": col_a,
                                                "col_b": col_b,
                                                "effect_size": cramers_v,
                                                "metric_name": "cramers_v",
                                                "p_value": round(p_value, 4)
                                                if p_value is not None
                                                else None,
                                                "sample_size": actual_sample_size,
                                            }
                                        )
                        except Exception:
                            pass

                    # Collect from Categorical ↔ Numeric
                    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                        try:
                            for cat_col in categorical_cols:
                                if (
                                    sample_df[cat_col].nunique()
                                    > categorical_cardinality_threshold
                                ):
                                    continue

                                for num_col in numeric_cols:
                                    eta = _compute_correlation_ratio(
                                        sample_df[cat_col], sample_df[num_col]
                                    )

                                    if pd.notna(eta):
                                        top_relationships.append(
                                            {
                                                "relationship_type": "cn",
                                                "col_a": cat_col,
                                                "col_b": num_col,
                                                "effect_size": eta,
                                                "metric_name": "eta",
                                                "p_value": None,
                                                "sample_size": actual_sample_size,
                                            }
                                        )
                        except Exception:
                            pass

                    if top_relationships:
                        top_df = pd.DataFrame(top_relationships)
                        top_df = top_df.sort_values(
                            "effect_size", ascending=False
                        ).head(50)

                        # Format for display
                        display_df = top_df.copy()
                        display_df.columns = [
                            "Type",
                            "Column A",
                            "Column B",
                            "Effect Size",
                            "Metric",
                            "p-value",
                            "Sample Size",
                        ]

                        st.dataframe(
                            display_df, use_container_width=True, hide_index=True
                        )
                    else:
                        st.info("No relationships found to display.")

                except Exception as e:
                    st.error(f"Error compiling top relationships: {e}")


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

    # Feature column selection
    try:
        schema_df = fetch_schema_summary(table_names=["feature_snapshots"])
        non_trainable_columns = [
            "record_id",
            "snapshot_id",
            "computed_at",
            "user_id",
            "experimental_signals",
        ]

        # Filter to numeric columns only (exclude JSONB, timestamps, IDs)
        numeric_types = [
            "integer",
            "bigint",
            "smallint",
            "real",
            "double precision",
            "numeric",
        ]
        available_columns = schema_df[
            (schema_df["table_name"] == "feature_snapshots")
            & (schema_df["data_type"].isin(numeric_types))
            & (~schema_df["column_name"].isin(non_trainable_columns))
        ]["column_name"].tolist()

        # Default feature columns (matching DataLoader.FEATURE_COLUMNS)
        default_feature_columns = [
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
        ]
        # Use defaults if available, otherwise use all available columns
        default_selection = [
            col for col in default_feature_columns if col in available_columns
        ] or available_columns

        # Initialize session state for feature columns if not set
        if "selected_feature_columns" not in st.session_state:
            st.session_state.selected_feature_columns = default_selection

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_columns = st.multiselect(
                "Feature Columns",
                options=available_columns,
                default=st.session_state.selected_feature_columns,
                help="Select which feature columns to use for training",
                key="feature_columns_multiselect",
            )
        with col2:
            if st.button("Reset to Defaults", help="Reset to default feature columns"):
                st.session_state.selected_feature_columns = default_selection
                st.session_state.feature_columns_multiselect = default_selection
                st.rerun()

        # Update session state
        st.session_state.selected_feature_columns = selected_columns

        # Show summary
        if selected_columns:
            st.caption(
                f"Selected {len(selected_columns)} of "
                f"{len(available_columns)} feature columns"
            )
            with st.expander("View Selected Columns"):
                st.write(", ".join(sorted(selected_columns)))
        else:
            st.warning(
                "No feature columns selected. Please select at least one column."
            )

    except Exception as e:
        st.error(f"Error loading feature columns: {e}")
        selected_columns = None

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

    with st.expander("Advanced Hyperparameters"):
        ah1, ah2 = st.columns(2)
        with ah1:
            n_estimators = st.slider(
                "n_estimators",
                min_value=50,
                max_value=500,
                value=100,
                step=25,
                help="Number of boosting rounds",
            )
            learning_rate = st.slider(
                "Learning Rate",
                min_value=0.01,
                max_value=0.3,
                value=0.1,
                step=0.01,
                format="%.2f",
                help="Step size shrinkage",
            )
            min_child_weight = st.slider(
                "min_child_weight",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
            )
            subsample = st.slider(
                "subsample",
                min_value=0.5,
                max_value=1.0,
                value=1.0,
                step=0.1,
                format="%.1f",
            )
            colsample_bytree = st.slider(
                "colsample_bytree",
                min_value=0.5,
                max_value=1.0,
                value=1.0,
                step=0.1,
                format="%.1f",
            )
        with ah2:
            gamma = st.slider(
                "gamma",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.1,
                format="%.1f",
            )
            reg_alpha = st.slider(
                "reg_alpha",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                format="%.1f",
            )
            reg_lambda = st.slider(
                "reg_lambda",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                format="%.1f",
            )
            early_stopping_rounds = st.number_input(
                "Early Stopping Rounds",
                min_value=0,
                max_value=50,
                value=0,
                step=5,
                help="0 = disabled. Stop if no improvement for N rounds.",
            )
            if early_stopping_rounds == 0:
                early_stopping_rounds = None

    with st.expander("Hyperparameter Tuning"):
        tuning_enabled = st.checkbox(
            "Enable tuning",
            value=False,
            help="Run Optuna to search for best hyperparameters before training.",
        )
        tune_col1, tune_col2 = st.columns(2)
        with tune_col1:
            tuning_n_trials = st.slider(
                "Trials",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                disabled=not tuning_enabled,
            )
            tuning_timeout = st.slider(
                "Timeout (min)",
                min_value=5,
                max_value=120,
                value=30,
                step=5,
                disabled=not tuning_enabled,
            )
        with tune_col2:
            tuning_metric = st.selectbox(
                "Optimize",
                options=["pr_auc", "roc_auc", "f1"],
                index=0,
                disabled=not tuning_enabled,
            )
        if tuning_enabled:
            st.caption("Training may take longer. Best params logged to MLflow.")

    train_clicked = st.button(
        "Start Training", type="primary", disabled=not selected_columns
    )

    if train_clicked:
        spinner_msg = (
            "Running tuning, then training..."
            if tuning_enabled
            else "Training model... This may take a moment."
        )
        timeout_sec = (tuning_timeout + 60) if tuning_enabled else 300
        with st.spinner(spinner_msg):
            try:
                import requests

                payload = {
                    "max_depth": max_depth,
                    "training_window_days": training_window,
                    "selected_feature_columns": selected_columns,
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "min_child_weight": min_child_weight,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "gamma": gamma,
                    "reg_alpha": reg_alpha,
                    "reg_lambda": reg_lambda,
                    "tuning_config": {
                        "enabled": tuning_enabled,
                        "n_trials": tuning_n_trials,
                        "timeout_minutes": tuning_timeout,
                        "metric": tuning_metric,
                    },
                }
                if early_stopping_rounds is not None:
                    payload["early_stopping_rounds"] = early_stopping_rounds
                response = requests.post(
                    f"{API_BASE_URL}/train",
                    json=payload,
                    timeout=timeout_sec,
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

        st.markdown("**Run details**")
        for _, row in runs_df.iterrows():
            run_id = row["Run ID"]
            pr_auc = row.get("PR-AUC", "")
            label = f"Run `{run_id}` — PR-AUC {pr_auc}"
            with st.expander(label):
                details = get_run_details(run_id)
                c1, c2 = st.columns(2)
                with c1:
                    if details["params"]:
                        st.markdown("**Parameters**")
                        st.json(details["params"])
                with c2:
                    if details["metrics"]:
                        st.markdown("**Metrics**")
                        st.json(details["metrics"])
                arts = get_run_artifacts(run_id)
                for a in arts:
                    if a["is_dir"]:
                        continue
                    path = a["path"]
                    if path.endswith("confusion_matrix.png"):
                        local = fetch_artifact_path(run_id, path)
                        if local:
                            st.image(local, caption="Confusion matrix")
                    elif path.endswith("feature_importance_plot.png"):
                        local = fetch_artifact_path(run_id, path)
                        if local:
                            st.image(local, caption="Feature importance")
                if arts:
                    st.markdown("**Artifacts**")
                    for a in arts:
                        if not a["is_dir"]:
                            st.caption(a["path"])

        run_ids = runs_df["Run ID"].tolist()

        st.markdown("**Compare runs**")
        compare_ids = st.multiselect(
            "Select runs to compare",
            options=run_ids,
            default=run_ids[:2] if len(run_ids) >= 2 else run_ids,
            help="Choose 2+ runs for side-by-side comparison",
        )
        if len(compare_ids) >= 2:
            compare_clicked = st.button("Compare selected")
            if compare_clicked:
                details_list = [get_run_details(rid) for rid in compare_ids]
                metric_keys = ["precision", "recall", "pr_auc", "f1", "roc_auc"]
                rows = []
                for k in metric_keys:
                    row = {"metric": k}
                    for i, rid in enumerate(compare_ids):
                        v = details_list[i].get("metrics", {}).get(k)
                        row[rid[:8]] = v if v is not None else ""
                    rows.append(row)
                comp_df = pd.DataFrame(rows).set_index("metric")
                st.dataframe(comp_df, use_container_width=True, hide_index=False)
                fig = go.Figure()
                for i, rid in enumerate(compare_ids):
                    vals = [
                        details_list[i].get("metrics", {}).get(k) for k in metric_keys
                    ]
                    fig.add_trace(go.Bar(name=rid[:12], x=metric_keys, y=vals))
                fig.update_layout(
                    barmode="group",
                    title="Metrics comparison",
                    xaxis_title="Metric",
                    yaxis_title="Value",
                )
                st.plotly_chart(fig, use_container_width=True)
                all_params = set()
                for d in details_list:
                    all_params.update(d.get("params", {}).keys())
                diffs = []
                for p in sorted(all_params):
                    vals = [str(d.get("params", {}).get(p, "")) for d in details_list]
                    if len(set(vals)) > 1:
                        diffs.append((p, vals))
                if diffs:
                    st.markdown("**Config diffs**")
                    for p, vals in diffs:
                        parts = " | ".join(
                            f"{rid[:8]}: {v}" for rid, v in zip(compare_ids, vals)
                        )
                        st.caption(f"**{p}**: {parts}")

        st.markdown("---")

        # Promote to production
        st.markdown("**Promote to Production**")
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
