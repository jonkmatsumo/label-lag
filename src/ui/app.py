"""ACH Risk Inspector Dashboard.

A Streamlit dashboard for fraud risk analysis with two modes:
- Live Scoring: Real-time transaction evaluation via API
- Historical Analytics: Analysis of historical data from database

NOTE: This service is isolated and does NOT import from src.model or src.generator.
"""

import os

import streamlit as st

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

    # Placeholder content
    st.info("🚧 Live scoring interface coming soon...")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Input")
        st.text_input("User ID", placeholder="user_abc123")
        st.number_input("Amount", min_value=0.01, value=100.00, step=0.01)
        st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD", "AUD"])
        st.text_input("Transaction ID", placeholder="txn_xyz789")
        st.button("Evaluate Risk", type="primary")

    with col2:
        st.subheader("Risk Assessment")
        st.markdown("*Submit a transaction to see results*")


def render_analytics() -> None:
    """Render the Historical Analytics (DB) page.

    This page provides analytics and visualizations based on
    historical transaction data stored in the database.
    """
    st.header("Historical Analytics")
    st.markdown("Analyze historical transaction patterns and fraud metrics.")

    # Placeholder content
    st.info("🚧 Analytics dashboard coming soon...")

    # Placeholder metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Transactions", value="--", delta=None)

    with col2:
        st.metric(label="Fraud Rate", value="--", delta=None)

    with col3:
        st.metric(label="Avg Risk Score", value="--", delta=None)

    with col4:
        st.metric(label="Blocked Amount", value="--", delta=None)

    # Placeholder for charts
    st.subheader("Score Distribution")
    st.markdown("*Connect to database to load charts*")

    st.subheader("Fraud Trends")
    st.markdown("*Connect to database to load charts*")


def main() -> None:
    """Main application entry point."""
    # Sidebar navigation
    st.sidebar.title("🔍 ACH Risk Inspector")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        options=["Live Scoring (API)", "Historical Analytics (DB)"],
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
    else:
        render_analytics()


if __name__ == "__main__":
    main()
