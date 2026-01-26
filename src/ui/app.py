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
    fetch_backtest_results,
    fetch_daily_stats,
    fetch_feature_sample,
    fetch_fraud_summary,
    fetch_heuristic_suggestions,
    fetch_overview_metrics,
    fetch_recent_alerts,
    fetch_rules,
    fetch_schema_summary,
    fetch_shadow_comparison,
    fetch_transaction_details,
    predict_risk,
    sandbox_evaluate,
)
from ui.mlflow_utils import (
    check_mlflow_connection,
    check_promotion_thresholds,
    deploy_model,
    fetch_artifact_path,
    get_cv_fold_metrics,
    get_experiment_runs,
    get_model_versions,
    get_production_model_version,
    get_run_artifacts,
    get_run_details,
    get_running_experiments,
    get_split_manifest,
    get_tuning_trials,
    get_version_details,
    promote_to_production,
    promote_to_staging,
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
        run_id = v.get("run_id", "")

        # Build label with indicators
        label = f"v{version}"
        indicators = []
        if stage == "Production":
            indicators.append("PRODUCTION")
        if live_model and live_model == f"v{version}":
            indicators.append("LIVE")
        if indicators:
            label += f" ({', '.join(indicators)})"
        elif stage and stage != "None":
            label += f" ({stage})"

        options.append(
            {
                "label": label,
                "version": version,
                "stage": stage,
                "run_id": run_id,
            }
        )

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
            st.session_state["selected_model_version"] = selected["version"]

            with st.expander("Version details"):
                det = get_version_details(version=selected["version"])
                if det:
                    st.markdown(f"**Run ID:** `{det.get('run_id', '—')}`")
                    m = det.get("metrics", {})
                    parts = []
                    if m.get("pr_auc") is not None:
                        parts.append(f"PR-AUC: {m['pr_auc']:.4f}")
                    if m.get("f1") is not None:
                        parts.append(f"F1: {m['f1']:.4f}")
                    if parts:
                        st.caption(" · ".join(parts))
                    if selected["stage"] == "Production":
                        st.success("Production model")


def _render_cv_fold_metrics(run_id: str) -> None:
    """Render CV fold metrics visualization for a run.

    Args:
        run_id: MLflow run ID.
    """
    cv_metrics = get_cv_fold_metrics(run_id)
    if not cv_metrics:
        return

    with st.expander("**CV Fold Metrics**", expanded=False):
        st.markdown(
            "Cross-validation metrics across folds. Boxplots show distribution, "
            "mean ± std shown as summary."
        )

        # Key metrics to display prominently
        key_metrics = ["precision", "recall", "pr_auc", "f1", "roc_auc"]
        other_metrics = [m for m in cv_metrics.keys() if m not in key_metrics]

        # Display key metrics with boxplots
        for metric_name in key_metrics:
            if metric_name not in cv_metrics:
                continue

            fold_values = cv_metrics[metric_name]
            if not fold_values:
                continue

            # Calculate summary stats
            mean_val = np.mean(fold_values)
            std_val = np.std(fold_values)
            min_val = np.min(fold_values)
            max_val = np.max(fold_values)

            st.markdown(f"**{metric_name.replace('_', ' ').title()}**")
            col1, col2 = st.columns([2, 1])

            with col1:
                # Boxplot
                fig = go.Figure()
                fig.add_trace(
                    go.Box(
                        y=fold_values,
                        name=metric_name,
                        boxmean="sd",  # Show mean and std
                    )
                )
                fig.update_layout(
                    title=f"{metric_name} across folds",
                    yaxis_title="Value",
                    height=200,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.metric("Mean", f"{mean_val:.4f}")
                st.metric("Std", f"{std_val:.4f}")
                st.metric("Min", f"{min_val:.4f}")
                st.metric("Max", f"{max_val:.4f}")

        # Display other metrics in a compact table
        if other_metrics:
            st.markdown("**Other Metrics**")
            summary_data = []
            for metric_name in sorted(other_metrics):
                fold_values = cv_metrics[metric_name]
                if fold_values:
                    summary_data.append(
                        {
                            "Metric": metric_name.replace("_", " ").title(),
                            "Mean": f"{np.mean(fold_values):.4f}",
                            "Std": f"{np.std(fold_values):.4f}",
                            "Min": f"{np.min(fold_values):.4f}",
                            "Max": f"{np.max(fold_values):.4f}",
                        }
                    )
            if summary_data:
                st.dataframe(
                    pd.DataFrame(summary_data),
                    use_container_width=True,
                    hide_index=True,
                )


def _render_split_summary(run_id: str) -> None:
    """Render split manifest summary for a run.

    Args:
        run_id: MLflow run ID.
    """
    manifest = get_split_manifest(run_id)
    if not manifest:
        return

    with st.expander("**Split Summary**", expanded=False):
        st.markdown("Train/test split configuration and statistics.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Training Set**")
            st.metric("Size", f"{manifest.get('train_size', 0):,}")
            st.metric(
                "Fraud Rate",
                f"{manifest.get('train_fraud_rate', 0.0):.2%}",
            )
            if manifest.get("train_unique_users"):
                st.metric(
                    "Unique Users",
                    f"{manifest.get('train_unique_users', 0):,}",
                )
            if manifest.get("train_time_range"):
                tr = manifest["train_time_range"]
                st.caption(f"Time Range: {tr.get('min', '')} to {tr.get('max', '')}")

        with col2:
            st.markdown("**Test Set**")
            st.metric("Size", f"{manifest.get('test_size', 0):,}")
            st.metric(
                "Fraud Rate",
                f"{manifest.get('test_fraud_rate', 0.0):.2%}",
            )
            if manifest.get("test_unique_users"):
                st.metric(
                    "Unique Users",
                    f"{manifest.get('test_unique_users', 0):,}",
                )
            if manifest.get("test_time_range"):
                tr = manifest["test_time_range"]
                st.caption(f"Time Range: {tr.get('min', '')} to {tr.get('max', '')}")

        # Strategy and reproducibility info
        st.markdown("**Configuration**")
        strategy = manifest.get("strategy", "unknown")
        seed = manifest.get("seed", "unknown")
        cutoff = manifest.get("training_cutoff_date", "unknown")
        st.caption(f"Strategy: {strategy} | Seed: {seed} | Cutoff: {cutoff}")

        if manifest.get("manifest_hash"):
            hash_val = manifest["manifest_hash"]
            st.caption(f"Manifest Hash: `{hash_val[:16]}...`")

        # Fold assignments if available
        fold_assignments = manifest.get("fold_assignments")
        if fold_assignments:
            st.markdown("**Fold Assignments**")
            st.caption(f"{len(fold_assignments)} folds configured")


def _render_tuning_trials(run_id: str) -> None:
    """Render tuning trials visualization for a run.

    Args:
        run_id: MLflow run ID.
    """
    trials_df = get_tuning_trials(run_id)
    if trials_df is None or len(trials_df) == 0:
        return

    details = get_run_details(run_id)
    tags = details.get("tags", {})
    selected_trial = tags.get("tuning.selected_trial", "best")
    selection_type = tags.get("tuning.selection_type", "auto")

    with st.expander("**Tuning Trials**", expanded=False):
        st.markdown(
            f"Hyperparameter tuning results. "
            f"Selected trial: **{selected_trial}** ({selection_type})"
        )

        # Find best trial (highest value)
        if "value" in trials_df.columns:
            best_trial_num = trials_df.loc[trials_df["value"].idxmax(), "trial"]
        else:
            best_trial_num = None

        # Display sortable table
        display_df = trials_df.copy()
        # Highlight selected/best trial
        if best_trial_num is not None:
            display_df["is_best"] = display_df["trial"] == best_trial_num
        if selected_trial != "best" and selected_trial.isdigit():
            display_df["is_selected"] = display_df["trial"] == int(selected_trial)

        # Show pruning summary
        if "state" in trials_df.columns:
            pruned = trials_df["state"].str.contains("PRUNED", na=False)
            completed = trials_df["state"].str.contains("COMPLETE", na=False)
            pruned_count = int(pruned.sum())
            completed_count = int(completed.sum())
            if pruned_count > 0:
                st.info(
                    f"**{pruned_count}/{len(trials_df)} trials pruned early** "
                    f"({completed_count} completed)"
                )

        # Select key columns for display
        key_cols = ["trial", "value", "state"]
        param_cols = [c for c in display_df.columns if c.startswith("params_")]
        # Show top 3 most variable params
        if param_cols:
            # Calculate variance for each param
            param_vars = {}
            for col in param_cols:
                if col in display_df.columns:
                    param_vars[col] = display_df[col].std()
            top_params = sorted(param_vars.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]
            display_cols = key_cols + [p[0] for p in top_params]
        else:
            display_cols = key_cols

        available_cols = [c for c in display_cols if c in display_df.columns]
        st.dataframe(
            display_df[available_cols].sort_values(
                by="value" if "value" in display_df.columns else "trial",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Scatter plot: trial number vs metric value
        if "value" in trials_df.columns and "trial" in trials_df.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=trials_df["trial"],
                    y=trials_df["value"],
                    mode="markers",
                    name="Trial",
                    marker=dict(size=8),
                )
            )
            # Highlight best trial
            if best_trial_num is not None:
                best_value = trials_df.loc[
                    trials_df["trial"] == best_trial_num, "value"
                ].iloc[0]
                fig.add_trace(
                    go.Scatter(
                        x=[best_trial_num],
                        y=[best_value],
                        mode="markers",
                        name="Best",
                        marker=dict(size=12, color="red", symbol="star"),
                    )
                )
            # Highlight selected trial if different from best
            if selected_trial != "best" and selected_trial.isdigit():
                sel_num = int(selected_trial)
                if sel_num in trials_df["trial"].values:
                    sel_value = trials_df.loc[
                        trials_df["trial"] == sel_num, "value"
                    ].iloc[0]
                    fig.add_trace(
                        go.Scatter(
                            x=[sel_num],
                            y=[sel_value],
                            mode="markers",
                            name="Selected",
                            marker=dict(size=12, color="green", symbol="diamond"),
                        )
                    )

            fig.update_layout(
                title="Trial Performance",
                xaxis_title="Trial Number",
                yaxis_title="Metric Value",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)


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


def render_drift_panel() -> None:
    """Render the dataset drift monitoring panel."""
    from ui.data_service import _cached_fetch_drift_status

    st.subheader("Dataset Drift Monitoring")

    # Fetch status with caching
    drift_status = _cached_fetch_drift_status()

    if drift_status is None:
        st.info("Drift monitoring unavailable. API may be offline.")
        return

    if drift_status.get("error"):
        st.warning(f"Drift check incomplete: {drift_status['error']}")
        return

    # Status badge
    status = drift_status.get("status", "unknown")
    if status == "ok":
        st.success("✅ No Drift Detected")
    elif status == "warn":
        st.warning("⚠️ Slight Drift Detected - Monitor closely")
    elif status == "fail":
        st.error("🚨 Significant Drift Detected - Action required")
    else:
        st.info("Drift status unknown")

    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reference Size", drift_status.get("reference_size", 0))
    with col2:
        st.metric("Live Size", drift_status.get("live_size", 0))
    with col3:
        computed = drift_status.get("computed_at", "Unknown")
        if computed != "Unknown":
            # Format timestamp for display
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(computed.replace("Z", "+00:00"))
                formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"Last check: {formatted}")
            except Exception:
                st.caption(f"Last check: {computed[:19]}")
        else:
            st.caption("Last check: Unknown")

    # Top features table
    top_features = drift_status.get("top_features", [])
    if top_features:
        st.markdown("**Top Features by PSI**")
        df = pd.DataFrame(top_features)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Guidance
    with st.expander("About PSI Thresholds"):
        st.markdown(
            """
        - **PSI < 0.1**: No significant change
        - **0.1 ≤ PSI < 0.2**: Slight drift, monitor
        - **PSI ≥ 0.2**: Significant drift, investigate
        """
        )


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

    # --- Drift Monitoring Panel ---
    render_drift_panel()

    st.markdown("---")

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
        import requests

        spinner_msg = (
            "1. Loading data · 2. Training model · 3. Computing metrics · "
            "4. Logging artifacts"
        )
        if tuning_enabled:
            spinner_msg = "Running tuning, then " + spinner_msg
        timeout_sec = (tuning_timeout + 60) if tuning_enabled else 300
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

        with st.spinner(spinner_msg):
            start = time.time()
            try:
                response = requests.post(
                    f"{API_BASE_URL}/train",
                    json=payload,
                    timeout=timeout_sec,
                )
                elapsed = time.time() - start
                result = response.json()

                if result.get("success"):
                    st.success(
                        f"Training complete! Run ID: `{result.get('run_id')}` "
                        f"(elapsed {elapsed:.0f}s)"
                    )
                    st.balloons()
                else:
                    st.error(
                        f"Training failed: {result.get('error')} "
                        f"(elapsed {elapsed:.0f}s)"
                    )
            except requests.exceptions.RequestException as e:
                elapsed = time.time() - start
                st.error(f"API request failed: {e} (elapsed {elapsed:.0f}s)")

    st.markdown("---")

    # --- Section C: Model Registry ---
    st.subheader("Model Registry")

    # Show current production model
    prod_version = get_production_model_version()
    if prod_version:
        st.info(f"Current Production Model: Version {prod_version}")
    else:
        st.warning("No production model deployed yet.")

    # Auto-refresh toggle
    auto_refresh = st.checkbox(
        "Auto-refresh (5s)",
        value=st.session_state.get("auto_refresh", False),
        help="Automatically refresh when training runs are in progress",
    )
    st.session_state["auto_refresh"] = auto_refresh

    # Check for running experiments and auto-refresh if enabled
    if auto_refresh:
        running = get_running_experiments()
        if running:
            st.info(f"🔄 {len(running)} training run(s) in progress...")
            time.sleep(5)
            st.rerun()

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
                # CV Fold Metrics section
                _render_cv_fold_metrics(run_id)
                # Split Summary section
                _render_split_summary(run_id)
                # Tuning Trials section
                _render_tuning_trials(run_id)
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
                    elif path.endswith("model_card.md"):
                        local = fetch_artifact_path(run_id, path)
                        if local:
                            try:
                                with open(local, encoding="utf-8") as f:
                                    raw = f.read()
                                st.markdown("**Model card**")
                                st.markdown(raw)
                            except OSError:
                                st.caption(path)
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
                pr_aucs = [
                    details_list[i].get("metrics", {}).get("pr_auc")
                    for i in range(len(compare_ids))
                ]
                f1s = [
                    details_list[i].get("metrics", {}).get("f1")
                    for i in range(len(compare_ids))
                ]
                valid = [
                    (p, f, rid)
                    for p, f, rid in zip(pr_aucs, f1s, compare_ids)
                    if p is not None and f is not None
                ]
                if len(valid) >= 2:
                    px_, fx_, tx_ = zip(*valid)
                    fig2 = go.Figure(
                        go.Scatter(
                            x=px_,
                            y=fx_,
                            text=[t[:12] for t in tx_],
                            mode="markers+text",
                            textposition="top center",
                        )
                    )
                    fig2.update_layout(
                        title="Tradeoff: PR-AUC vs F1",
                        xaxis_title="PR-AUC",
                        yaxis_title="F1",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
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

        # Model promotion workflow
        st.markdown("**Model Promotion**")
        selected_run = st.selectbox(
            "Select Run ID",
            options=run_ids,
            index=0,
            help="Choose a model run to promote",
        )

        if selected_run:
            # Get current stage of selected run
            versions = get_model_versions()
            selected_stage = None
            for v in versions:
                if v["run_id"] == selected_run:
                    selected_stage = v["stage"]
                    break

            # Load thresholds config
            import json
            from pathlib import Path

            thresholds_config = {}
            thresholds_path = Path("config/model_thresholds.json")
            if thresholds_path.exists():
                with open(thresholds_path) as f:
                    thresholds_config = json.load(f)

            production_thresholds = thresholds_config.get("production_thresholds", {})

            col1, col2 = st.columns(2)

            with col1:
                # Promote to Staging
                if selected_stage != "Staging" and selected_stage != "Production":
                    if st.button("Promote to Staging", type="secondary"):
                        with st.spinner("Promoting to Staging..."):
                            result = promote_to_staging(selected_run)
                        if result["success"]:
                            st.success(result["message"])
                            st.rerun()
                        else:
                            st.error(result["message"])

            with col2:
                # Approve for Production (requires Staging)
                if selected_stage == "Staging":
                    # Check thresholds
                    if production_thresholds:
                        passed, failures = check_promotion_thresholds(
                            selected_run, production_thresholds
                        )
                        if not passed:
                            st.warning(f"Threshold check failed: {', '.join(failures)}")
                            st.button(
                                "Approve for Production",
                                disabled=True,
                                help="Fix threshold failures first",
                            )
                        else:
                            if st.button("Approve for Production", type="secondary"):
                                with st.spinner("Approving for Production..."):
                                    result = promote_to_production(
                                        selected_run, thresholds=production_thresholds
                                    )
                                if result["success"]:
                                    st.success(result["message"])
                                    st.rerun()
                                else:
                                    st.error(result["message"])
                    else:
                        if st.button("Approve for Production", type="secondary"):
                            with st.spinner("Approving for Production..."):
                                result = promote_to_production(selected_run)
                            if result["success"]:
                                st.success(result["message"])
                                st.rerun()
                            else:
                                st.error(result["message"])
                elif selected_stage == "Production":
                    # Check if model is already deployed (compare with live model)
                    api_health = check_api_health()
                    live_model = None
                    if api_health and api_health.get("model_loaded"):
                        live_model = api_health.get("version", "unknown")

                    # Get model version for selected run
                    selected_version = None
                    for v in versions:
                        if v["run_id"] == selected_run:
                            selected_version = f"v{v['version']}"
                            break

                    if live_model == selected_version:
                        st.success("✅ Model is live in production")
                    else:
                        st.info("⏳ Model approved but not yet deployed")
                        st.markdown("---")
                        if st.button("Deploy to Production", type="primary"):
                            # Show confirmation modal
                            with st.form(key="deploy_model_form"):
                                st.warning(
                                    f"**Deploy Model to Production**\n\n"
                                    f"This will route live traffic to model version "
                                    f"{selected_version}.\n\n"
                                    f"Current live model: {live_model or 'none'}"
                                )
                                actor = st.text_input(
                                    "Your name/email",
                                    key="deploy_actor",
                                    help="Required for audit trail",
                                )
                                deploy_reason = st.text_area(
                                    "Reason (optional)",
                                    key="deploy_reason",
                                    help="Optional reason for deployment",
                                )
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    deploy_confirm = st.form_submit_button(
                                        "Deploy", type="primary"
                                    )
                                with col_b:
                                    st.form_submit_button("Cancel")

                                if deploy_confirm and actor:
                                    with st.spinner("Deploying model..."):
                                        deploy_reason_val = (
                                            deploy_reason if deploy_reason else None
                                        )
                                        result = deploy_model(
                                            actor=actor, reason=deploy_reason_val
                                        )
                                    if result["success"]:
                                        st.success(result["message"])
                                        st.rerun()
                                    else:
                                        st.error(result["message"])
                                elif deploy_confirm and not actor:
                                    st.error("Please provide your name/email.")
                else:
                    st.info("Promote to Staging first")
    else:
        st.info("No experiment runs found. Train a model to see results here.")


def render_rule_inspector() -> None:
    """Render the Rule Inspector page.

    This page provides read-only inspection and testing of decision rules:
    - Sandbox: Deterministic rule testing
    - Shadow Metrics: Production vs shadow comparison
    - Backtest Results: Historical backtest viewer
    - Suggestions: Heuristic rule suggestions

    All features are read-only - no production modifications possible.
    """
    st.header("Rule Inspector")

    # Safety banner
    st.warning(
        "**Read-Only Inspection Mode** - No production changes possible. "
        "All operations are safe for exploration."
    )

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Rule Management",
            "Sandbox (Testing)",
            "Shadow Metrics (Read-Only)",
            "Backtest Results (Read-Only)",
            "Suggestions (Read-Only)",
        ]
    )

    # --- Tab 1: Rule Management ---
    with tab1:
        _render_rule_management_tab()

    # --- Tab 2: Sandbox ---
    with tab2:
        _render_sandbox_tab()

    # --- Tab 3: Shadow Metrics ---
    with tab3:
        _render_shadow_metrics_tab()

    # --- Tab 4: Backtest Results ---
    with tab4:
        _render_backtest_tab()

    # --- Tab 5: Suggestions ---
    with tab5:
        _render_suggestions_tab()


def _render_rule_management_tab() -> None:
    """Render the Rule Management tab for draft rules."""
    from data_service import fetch_draft_rules, publish_rule

    st.subheader("Rule Management")
    st.markdown(
        "Manage draft rules: view status, approve, and publish to production."
    )

    # Status filter
    status_filter = st.selectbox(
        "Filter by Status",
        options=["All", "draft", "pending_review", "approved", "active"],
        index=0,
    )

    # Fetch all rules first to calculate pending publish count (unfiltered)
    with st.spinner("Loading rules..."):
        all_rules_data = fetch_draft_rules(status=None)
        if all_rules_data is None:
            st.error("Failed to load rules. Is the API server running?")
            return

        all_rules = all_rules_data.get("rules", [])
        approved_rules = [r for r in all_rules if r.get("status") == "approved"]

        # Show pending publish count (from all rules, not filtered)
        if approved_rules:
            st.info(f"**{len(approved_rules)} approved rule(s) pending publish**")

        # Fetch filtered rules for display
        status_param = None if status_filter == "All" else status_filter
        rules_data = fetch_draft_rules(status=status_param)

    if rules_data is None:
        st.error("Failed to load filtered rules. Is the API server running?")
        return

    rules = rules_data.get("rules", [])
    total = rules_data.get("total", 0)

    if not rules:
        st.info("No rules found.")
        return

    # Display rules in a table
    st.markdown(f"**{total} rule(s) found**")

    for rule in rules:
        rule_id = rule.get("rule_id", "")
        status = rule.get("status", "")
        field = rule.get("field", "")
        op = rule.get("op", "")
        value = rule.get("value", "")
        action = rule.get("action", "")
        score = rule.get("score", "")
        severity = rule.get("severity", "")
        reason = rule.get("reason", "")

        with st.expander(
            f"{rule_id} - {field} {op} {value} [{status}]", expanded=False
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Rule ID:** `{rule_id}`")
                st.markdown(f"**Status:** {status}")
                st.markdown(f"**Field:** {field}")
                st.markdown(f"**Operator:** {op}")
                st.markdown(f"**Value:** {value}")
                st.markdown(f"**Action:** {action}")
                if score:
                    st.markdown(f"**Score:** {score}")
                st.markdown(f"**Severity:** {severity}")
                if reason:
                    st.markdown(f"**Reason:** {reason}")

                # Show approval signals for pending_review rules
                if status == "pending_review":
                    from data_service import fetch_approval_signals

                    with st.spinner("Loading approval signals..."):
                        signals_data = fetch_approval_signals(rule_id)

                    if signals_data:
                        st.markdown("---")
                        st.markdown("### 📊 Approval Signals")

                        summary = signals_data.get("summary", {})
                        signals = signals_data.get("signals", [])
                        partial = signals_data.get("partial", False)
                        unavailable = signals_data.get("unavailable_signals", [])

                        # Group signals by severity
                        risk_signals = [s for s in signals if s.get("severity") == "risk"]
                        warning_signals = [
                            s for s in signals if s.get("severity") == "warning"
                        ]
                        info_signals = [s for s in signals if s.get("severity") == "info"]

                        # Risk signals (always visible)
                        if risk_signals:
                            st.error(f"🔴 **{len(risk_signals)} Risk Signal(s)**")
                            for signal in risk_signals:
                                st.markdown(
                                    f"- **{signal.get('label', 'Unknown')}**: "
                                    f"{signal.get('description', '')}"
                                )

                        # Warning signals (always visible)
                        if warning_signals:
                            st.warning(f"⚠️ **{len(warning_signals)} Warning(s)**")
                            for signal in warning_signals:
                                st.markdown(
                                    f"- **{signal.get('label', 'Unknown')}**: "
                                    f"{signal.get('description', '')}"
                                )

                        # Info signals (collapsed by default)
                        if info_signals:
                            with st.expander(
                                f"ℹ️ {len(info_signals)} Info Signal(s)", expanded=False
                            ):
                                for signal in info_signals:
                                    value = signal.get("value")
                                    if value is not None:
                                        st.markdown(
                                            f"- **{signal.get('label', 'Unknown')}**: "
                                            f"{value} - {signal.get('description', '')}"
                                        )
                                    else:
                                        st.markdown(
                                            f"- **{signal.get('label', 'Unknown')}**: "
                                            f"{signal.get('description', '')}"
                                        )

                        # Partial/unavailable notice
                        if partial:
                            st.caption(
                                f"⚠️ Some signals unavailable: {', '.join(unavailable)}"
                            )

                        # No issues case
                        if (
                            summary.get("risk_count", 0) == 0
                            and summary.get("warning_count", 0) == 0
                        ):
                            st.success("✅ No issues detected")
                    elif signals_data is None:
                        st.caption("⚠️ Could not load approval signals")

            with col2:
                # Status badge
                if status == "draft":
                    st.info("📝 Draft")
                elif status == "pending_review":
                    st.warning("⏳ Pending Review")
                elif status == "approved":
                    st.info("✅ Approved")
                elif status == "active":
                    st.success("🟢 Active")
                elif status == "shadow":
                    st.info("👻 Shadow")
                elif status == "disabled":
                    st.error("🚫 Disabled")
                else:
                    st.caption(f"Status: {status}")

                # Publish button for approved rules
                if status == "approved":
                    st.markdown("---")
                    if st.button(
                        "Publish to Production",
                        key=f"publish_{rule_id}",
                        type="primary",
                    ):
                        # Show confirmation modal
                        with st.form(key=f"publish_form_{rule_id}"):
                            st.warning(
                                f"**Publish Rule to Production**\n\n"
                                f"This will make rule `{rule_id}` effective for "
                                f"all live transactions."
                            )
                            actor = st.text_input(
                                "Your name/email",
                                key=f"actor_{rule_id}",
                                help="Required for audit trail",
                            )
                            publish_reason = st.text_area(
                                "Reason (optional)",
                                key=f"reason_{rule_id}",
                                help="Optional reason for publishing",
                            )
                            col_a, col_b = st.columns(2)
                            with col_a:
                                publish_confirm = st.form_submit_button(
                                    "Publish", type="primary"
                                )
                            with col_b:
                                st.form_submit_button("Cancel")

                            if publish_confirm and actor:
                                with st.spinner("Publishing rule..."):
                                    publish_reason_val = (
                                        publish_reason if publish_reason else None
                                    )
                                    result = publish_rule(
                                        rule_id=rule_id,
                                        actor=actor,
                                        reason=publish_reason_val,
                                    )
                                if result:
                                    st.success(
                                        f"Rule `{rule_id}` published successfully!"
                                    )
                                    st.rerun()
                                else:
                                    st.error("Failed to publish rule.")
                            elif publish_confirm and not actor:
                                st.error("Please provide your name/email.")


def _render_sandbox_tab() -> None:
    """Render the Sandbox testing tab."""
    st.subheader("Rule Sandbox")
    st.markdown(
        "Test rule evaluation against arbitrary feature inputs. "
        "**No database writes, no model inference, no production changes.**"
    )

    st.info(
        "**SANDBOX MODE** - This is a pure function evaluation. "
        "Custom rulesets are ephemeral and not saved."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Feature Inputs")

        # Feature sliders/inputs
        velocity_24h = st.slider(
            "velocity_24h",
            min_value=0,
            max_value=50,
            value=3,
            help="Transaction count in last 24 hours",
        )

        amount_to_avg_ratio = st.slider(
            "amount_to_avg_ratio_30d",
            min_value=0.0,
            max_value=20.0,
            value=1.5,
            step=0.1,
            help="Transaction amount vs 30-day average ratio",
        )

        balance_volatility = st.slider(
            "balance_volatility_z_score",
            min_value=-5.0,
            max_value=5.0,
            value=0.0,
            step=0.1,
            help="Balance volatility z-score",
        )

        bank_connections = st.slider(
            "bank_connections_24h",
            min_value=0,
            max_value=30,
            value=1,
            help="Bank connections in last 24 hours",
        )

        merchant_risk = st.slider(
            "merchant_risk_score",
            min_value=0,
            max_value=100,
            value=30,
            help="Merchant risk score",
        )

        has_history = st.checkbox("has_history", value=True, help="User has history")

        transaction_amount = st.number_input(
            "transaction_amount",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help="Transaction amount",
        )

        st.markdown("---")

        base_score = st.slider(
            "Base Score",
            min_value=1,
            max_value=99,
            value=50,
            help="Base score before rule application",
        )

    with col2:
        st.markdown("#### Ruleset Selection")

        ruleset_source = st.radio(
            "Ruleset Source",
            options=["Production Ruleset", "Custom JSON"],
            index=0,
            help="Choose ruleset to evaluate",
        )

        custom_ruleset = None
        if ruleset_source == "Custom JSON":
            default_json = """{
  "version": "test_v1",
  "rules": [
    {
      "id": "high_velocity",
      "field": "velocity_24h",
      "op": ">",
      "value": 5,
      "action": "clamp_min",
      "score": 70,
      "severity": "medium",
      "reason": "High transaction velocity",
      "status": "active"
    }
  ]
}"""
            json_input = st.text_area(
                "Custom Ruleset JSON",
                value=default_json,
                height=300,
                help="Paste custom ruleset JSON (ephemeral - not saved)",
            )

            try:
                custom_ruleset = json.loads(json_input)
                st.success("Valid JSON")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                custom_ruleset = None
        else:
            # Show current production ruleset
            with st.expander("View Production Ruleset"):
                prod_rules = fetch_rules()
                if prod_rules:
                    st.json(prod_rules)
                else:
                    st.info("No production ruleset loaded.")

        st.markdown("---")

        evaluate_clicked = st.button("Evaluate Rules", type="primary")

    # Evaluation results
    if evaluate_clicked:
        features = {
            "velocity_24h": velocity_24h,
            "amount_to_avg_ratio_30d": amount_to_avg_ratio,
            "balance_volatility_z_score": balance_volatility,
            "bank_connections_24h": bank_connections,
            "merchant_risk_score": merchant_risk,
            "has_history": has_history,
            "transaction_amount": transaction_amount,
        }

        with st.spinner("Evaluating rules..."):
            result = sandbox_evaluate(features, base_score, custom_ruleset)

        if result is None:
            st.error("Evaluation failed. Is the API server running?")
        else:
            st.markdown("---")
            st.markdown("### Evaluation Results")

            # Score display
            final_score = result.get("final_score", 0)
            rejected = result.get("rejected", False)

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if rejected:
                    st.error(f"**Final Score: {final_score}** (REJECTED)")
                elif final_score >= 80:
                    st.error(f"**Final Score: {final_score}** (High Risk)")
                elif final_score >= 50:
                    st.warning(f"**Final Score: {final_score}** (Medium Risk)")
                else:
                    st.success(f"**Final Score: {final_score}** (Low Risk)")

            with col_b:
                st.metric("Base Score", base_score)

            with col_c:
                st.metric("Score Change", final_score - base_score)

            # Matched rules table
            matched_rules = result.get("matched_rules", [])
            if matched_rules:
                st.markdown("#### Matched Rules (Applied to Score)")
                rules_df = pd.DataFrame(matched_rules)
                st.dataframe(rules_df, use_container_width=True, hide_index=True)
            else:
                st.info("No rules matched.")

            # Shadow matched rules table
            shadow_matched = result.get("shadow_matched_rules", [])
            if shadow_matched:
                st.markdown("#### Shadow Rules Matched (Not Applied)")
                shadow_df = pd.DataFrame(shadow_matched)
                st.dataframe(shadow_df, use_container_width=True, hide_index=True)

            # Raw response
            with st.expander("View Raw Response"):
                st.json(result)


def _render_shadow_metrics_tab() -> None:
    """Render the Shadow Metrics tab."""
    st.subheader("Shadow Mode Metrics")
    st.markdown(
        "Compare production vs shadow rule performance. "
        "**Read-only view - no modifications possible.**"
    )

    # Date range selector
    from datetime import datetime as dt
    from datetime import timedelta

    col1, col2 = st.columns(2)

    with col1:
        default_start = dt.now() - timedelta(days=30)
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            help="Start of comparison period",
        )

    with col2:
        end_date = st.date_input(
            "End Date",
            value=dt.now(),
            help="End of comparison period",
        )

    fetch_clicked = st.button("Fetch Metrics", type="primary")

    if fetch_clicked:
        with st.spinner("Fetching shadow metrics..."):
            result = fetch_shadow_comparison(
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )

        if result is None:
            st.error("Failed to fetch metrics. Is the API server running?")
        else:
            rule_metrics = result.get("rule_metrics", [])
            total_requests = result.get("total_requests", 0)

            # Summary cards
            st.markdown("---")
            st.markdown("### Summary")

            col_a, col_b, col_c, col_d = st.columns(4)

            total_prod = sum(m.get("production_matches", 0) for m in rule_metrics)
            total_shadow = sum(m.get("shadow_matches", 0) for m in rule_metrics)
            total_overlap = sum(m.get("overlap_count", 0) for m in rule_metrics)

            with col_a:
                st.metric("Total Requests", total_requests)

            with col_b:
                st.metric("Production Matches", total_prod)

            with col_c:
                st.metric("Shadow Matches", total_shadow)

            with col_d:
                overlap_pct = (
                    (total_overlap / max(total_prod, 1)) * 100 if total_prod > 0 else 0
                )
                st.metric("Overlap %", f"{overlap_pct:.1f}%")

            # Per-rule metrics table
            if rule_metrics:
                st.markdown("### Per-Rule Metrics")
                metrics_df = pd.DataFrame(rule_metrics)

                # Select display columns
                display_cols = [
                    "rule_id",
                    "production_matches",
                    "shadow_matches",
                    "overlap_count",
                    "production_only_count",
                    "shadow_only_count",
                ]
                available_cols = [c for c in display_cols if c in metrics_df.columns]
                if available_cols:
                    st.dataframe(
                        metrics_df[available_cols],
                        use_container_width=True,
                        hide_index=True,
                    )

                # Bar chart comparison
                if len(metrics_df) > 0:
                    st.markdown("### Production vs Shadow Matches")
                    chart_data = metrics_df[
                        ["rule_id", "production_matches", "shadow_matches"]
                    ].melt(
                        id_vars=["rule_id"],
                        var_name="Match Type",
                        value_name="Count",
                    )
                    fig = px.bar(
                        chart_data,
                        x="rule_id",
                        y="Count",
                        color="Match Type",
                        barmode="group",
                        title="Rule Match Comparison",
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "No shadow metrics collected yet. "
                    "Shadow rules record matches when requests are processed."
                )


def _render_backtest_tab() -> None:
    """Render the Backtest Results tab."""
    st.subheader("Backtest Results")
    st.markdown(
        "Browse completed backtest results. "
        "**Read-only view - no backtests can be triggered from here.**"
    )

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        rule_id_filter = st.text_input(
            "Rule ID Filter",
            value="",
            help="Filter by specific rule ID (optional)",
        )

    with col2:
        limit = st.slider(
            "Max Results",
            min_value=10,
            max_value=100,
            value=50,
            help="Maximum results to display",
        )

    with col3:
        fetch_clicked = st.button("Fetch Results", type="primary")

    if fetch_clicked or "backtest_results" not in st.session_state:
        with st.spinner("Fetching backtest results..."):
            result = fetch_backtest_results(
                rule_id=rule_id_filter if rule_id_filter else None,
                limit=limit,
            )

        if result is None:
            st.error("Failed to fetch results. Is the API server running?")
            st.session_state.backtest_results = None
        else:
            st.session_state.backtest_results = result

    results = st.session_state.get("backtest_results")
    if results:
        results_list = results.get("results", [])
        total = results.get("total", 0)

        st.markdown(f"**{total} result(s) found**")

        if results_list:
            # Results table
            table_data = []
            for r in results_list:
                metrics = r.get("metrics", {})
                table_data.append(
                    {
                        "Job ID": r.get("job_id", ""),
                        "Rule ID": r.get("rule_id") or "All",
                        "Ruleset Version": r.get("ruleset_version", ""),
                        "Match Rate": f"{metrics.get('match_rate', 0) * 100:.1f}%",
                        "Total Records": metrics.get("total_records", 0),
                        "Completed At": r.get("completed_at", "")[:19],
                    }
                )

            results_df = pd.DataFrame(table_data)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # Detail view
            st.markdown("### Result Details")
            selected_job = st.selectbox(
                "Select Job ID for Details",
                options=[r.get("job_id", "") for r in results_list],
                help="View detailed metrics for a specific backtest",
            )

            if selected_job:
                selected_result = next(
                    (r for r in results_list if r.get("job_id") == selected_job), None
                )
                if selected_result:
                    metrics = selected_result.get("metrics", {})

                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Total Records", metrics.get("total_records", 0))
                    with col_b:
                        st.metric("Matched Count", metrics.get("matched_count", 0))
                    with col_c:
                        st.metric(
                            "Match Rate",
                            f"{metrics.get('match_rate', 0) * 100:.1f}%",
                        )
                    with col_d:
                        st.metric("Rejected Count", metrics.get("rejected_count", 0))

                    # Score distribution
                    score_dist = metrics.get("score_distribution", {})
                    if score_dist:
                        st.markdown("#### Score Distribution")
                        dist_df = pd.DataFrame(
                            [{"Range": k, "Count": v} for k, v in score_dist.items()]
                        )
                        fig = px.bar(
                            dist_df,
                            x="Range",
                            y="Count",
                            title="Score Distribution",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Error display
                    error = selected_result.get("error")
                    if error:
                        st.error(f"Backtest Error: {error}")

                    # Full details
                    with st.expander("View Full Details"):
                        st.json(selected_result)
        else:
            st.info(
                "No backtest results available. "
                "Backtests must be triggered via CLI or API."
            )


def _render_suggestions_tab() -> None:
    """Render the Suggestions tab."""
    st.subheader("Rule Suggestions")
    st.markdown(
        "Heuristic rule suggestions based on feature distribution analysis. "
        "**Preview only - no rules are created.**"
    )

    st.warning(
        "**PREVIEW ONLY** - These are suggestions based on statistical analysis. "
        "No rules are created or modified. Review carefully before implementing."
    )

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        field_options = [
            "",
            "velocity_24h",
            "amount_to_avg_ratio_30d",
            "balance_volatility_z_score",
        ]
        field_filter = st.selectbox(
            "Field Filter",
            options=field_options,
            help="Filter suggestions by field (optional)",
        )

    with col2:
        min_confidence = st.slider(
            "Min Confidence",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence threshold",
        )

    with col3:
        min_samples = st.number_input(
            "Min Samples",
            min_value=10,
            max_value=10000,
            value=100,
            step=50,
            help="Minimum samples required",
        )

    with col4:
        generate_clicked = st.button("Generate Suggestions", type="primary")

    if generate_clicked:
        with st.spinner("Analyzing feature distributions..."):
            result = fetch_heuristic_suggestions(
                field=field_filter if field_filter else None,
                min_confidence=min_confidence,
                min_samples=min_samples,
            )

        if result is None:
            st.error("Failed to generate suggestions. Is the API server running?")
        else:
            suggestions = result.get("suggestions", [])
            total = result.get("total", 0)

            st.markdown(f"**{total} suggestion(s) generated**")

            if suggestions:
                # Suggestions table
                table_data = []
                for s in suggestions:
                    table_data.append(
                        {
                            "Field": s.get("field", ""),
                            "Operator": s.get("operator", ""),
                            "Threshold": f"{s.get('threshold', 0):.2f}",
                            "Action": s.get("action", ""),
                            "Score": s.get("suggested_score", 0),
                            "Confidence": f"{s.get('confidence', 0) * 100:.0f}%",
                            "Reason": s.get("reason", ""),
                        }
                    )

                suggestions_df = pd.DataFrame(table_data)
                st.dataframe(suggestions_df, use_container_width=True, hide_index=True)

                # Evidence details
                st.markdown("### Evidence Details")
                for i, s in enumerate(suggestions):
                    evidence = s.get("evidence", {})
                    with st.expander(
                        f"{s.get('field')} {s.get('operator')} "
                        f"{s.get('threshold', 0):.2f}"
                    ):
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Mean", f"{evidence.get('mean', 0):.2f}")
                        with col_b:
                            st.metric("Std Dev", f"{evidence.get('std', 0):.2f}")
                        with col_c:
                            st.metric(
                                "Threshold Value", f"{evidence.get('value', 0):.2f}"
                            )
                        with col_d:
                            st.metric("Sample Count", evidence.get("sample_count", 0))

                        st.caption(f"Statistic: {evidence.get('statistic', 'N/A')}")

                st.markdown("---")
                st.info(
                    "To implement a suggestion, create a rule manually with the "
                    "suggested parameters via the API or ruleset configuration."
                )
            else:
                st.info(
                    "No suggestions generated. This could mean insufficient data "
                    "or no patterns meet the confidence threshold."
                )


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
            "Rule Inspector",
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
    elif page == "Model Lab":
        render_model_lab()
    else:
        render_rule_inspector()


if __name__ == "__main__":
    main()
