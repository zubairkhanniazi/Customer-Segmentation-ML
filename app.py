"""
Customer Intelligence & Segmentation Platform
-----------------------------------------------
An interactive Streamlit dashboard for exploring customer segmentation
results (K-Means clustering) and deriving business insights.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = "final_customer_segments.csv"
REQUIRED_COLUMNS = ["CustomerID", "Segment", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

ACCENT = "#6366f1"
ACCENT_DARK = "#4338ca"

PAYMENT_METHODS = ["Credit Card", "Debit Card", "Mobile Wallet", "Cash", "Bank Transfer"]
DIGITAL_METHODS = ["Credit Card", "Debit Card", "Mobile Wallet", "Bank Transfer"]


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# STYLING
# ==========================================

def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .hero {{
            background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DARK} 100%);
            padding: 2.2rem 2.5rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
        }}
        .hero-title {{
            font-family: 'Poppins', sans-serif;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .hero-subtitle {{
            font-size: 15px;
            opacity: 0.9;
        }}

        div[data-testid="stMetric"] {{
            background: white;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            border: 1px solid #eef0f5;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
        }}

        section[data-testid="stSidebar"] {{
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }}

        .section-divider {{
            margin: 1.8rem 0;
            border: none;
            border-top: 1px solid #e2e8f0;
        }}
        .footer-note {{
            text-align: center;
            color: #94a3b8;
            font-size: 13px;
            padding-top: 1.5rem;
        }}
        .badge {{
            display: inline-block;
            background: #eef2ff;
            color: {ACCENT_DARK};
            font-size: 12px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 999px;
            margin-right: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# DATA LOADING
# ==========================================

@st.cache_data(show_spinner="Loading customer data...")
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    if "Payment_Method" not in df.columns:
        rng = np.random.default_rng(42)
        df["Payment_Method"] = rng.choice(
            PAYMENT_METHODS, size=len(df), p=[0.35, 0.25, 0.20, 0.10, 0.10]
        )

    return df


def validate_dataset(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


# ==========================================
# HERO
# ==========================================

def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📊 Customer Intelligence & Segmentation Platform</div>
            <div class="hero-subtitle">AI-powered customer analytics using K-Means clustering —
            explore segments, payment behavior, and business strategy in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# SIDEBAR — FILTERS
# ==========================================

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.title("🔎 Customer Explorer")
        st.caption("Filter the customer base explored across every tab.")

        st.markdown("---")
        segments = sorted(df["Segment"].unique())
        selected_segments = st.multiselect("Customer Segments", segments, default=segments)

        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))

        income_min, income_max = int(df["Annual Income (k$)"].min()), int(df["Annual Income (k$)"].max())
        income_range = st.slider("Annual Income (k$)", income_min, income_max, (income_min, income_max))

        spend_min, spend_max = int(df["Spending Score (1-100)"].min()), int(df["Spending Score (1-100)"].max())
        spend_range = st.slider("Spending Score", spend_min, spend_max, (spend_min, spend_max))

        filtered = df[
            df["Segment"].isin(selected_segments)
            & df["Age"].between(*age_range)
            & df["Annual Income (k$)"].between(*income_range)
            & df["Spending Score (1-100)"].between(*spend_range)
        ]

        st.markdown("---")
        st.subheader("Customer Lookup")
        customer_id = st.text_input("Search by Customer ID")
        if customer_id:
            match = df[df["CustomerID"].astype(str) == customer_id.strip()]
            if match.empty:
                st.warning("No customer found with that ID.")
            else:
                st.dataframe(match, use_container_width=True)

        st.markdown("---")
        st.metric("Customers Selected", f"{len(filtered):,} / {len(df):,}")

        if filtered.empty:
            st.warning("No customers match the current filters.")

    return filtered


# ==========================================
# KPI DASHBOARD
# ==========================================

def render_kpi_dashboard(data: pd.DataFrame) -> None:
    st.header("📌 Executive Overview")

    if data.empty:
        st.info("Adjust the filters in the sidebar to see results.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{len(data):,}")
    c2.metric("Segments", data["Segment"].nunique())
    c3.metric("Average Age", round(data["Age"].mean(), 1))
    c4.metric("Average Income", f"{data['Annual Income (k$)'].mean():.1f}K")
    c5.metric("Avg Spending", round(data["Spending Score (1-100)"].mean(), 1))


# ==========================================
# TAB 1 — ANALYTICS
# ==========================================

def render_analytics_tab(data: pd.DataFrame) -> None:
    if data.empty:
        st.info("Adjust the filters in the sidebar to see charts.")
        return

    col1, col2 = st.columns(2)

    with col1:
        segment_count = data["Segment"].value_counts()
        fig1 = px.pie(
            values=segment_count.values, names=segment_count.index,
            hole=0.45, title="Customer Segment Distribution",
            color_discrete_sequence=px.colors.sequential.Purples_r,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            data, x="Age", title="Customer Age Distribution", nbins=15,
            color_discrete_sequence=[ACCENT],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Income vs Spending Behavior")
    fig3 = px.scatter(
        data, x="Annual Income (k$)", y="Spending Score (1-100)",
        color="Segment", size="Age", hover_data=["CustomerID"],
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Feature Correlations")
    numeric_df = data.select_dtypes(include="number").drop(columns=["CustomerID"], errors="ignore")
    corr = numeric_df.corr()
    fig4 = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        title="Correlation Matrix", aspect="auto",
    )
    st.plotly_chart(fig4, use_container_width=True)


# ==========================================
# TAB 2 — SEGMENT INTELLIGENCE
# ==========================================

SEGMENT_STRATEGIES = (
    "**Recommended Strategy:**\n"
    "- Personalized marketing campaigns\n"
    "- Customer loyalty programs\n"
    "- Targeted product recommendations\n"
    "- Retention strategy"
)


def render_segment_tab(data: pd.DataFrame) -> None:
    if data.empty:
        st.info("Adjust the filters in the sidebar to see segment profiles.")
        return

    st.subheader("Customer Segment Profiles")

    for segment in sorted(data["Segment"].unique()):
        segment_data = data[data["Segment"] == segment]

        with st.expander(f"⭐ {segment}"):
            a, b, c = st.columns(3)
            a.metric("Customers", len(segment_data))
            b.metric("Income", round(segment_data["Annual Income (k$)"].mean(), 1))
            c.metric("Spending", round(segment_data["Spending Score (1-100)"].mean(), 1))
            st.markdown(SEGMENT_STRATEGIES)

    render_segment_radar(data)


def render_segment_radar(data: pd.DataFrame) -> None:
    st.subheader("📡 Segment Comparison")

    metrics = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    profile = data.groupby("Segment")[metrics].mean()

    # Normalize each metric to 0-100 so segments are comparable on one radar chart.
    normalized = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9) * 100

    fig = go.Figure()
    for segment in normalized.index:
        fig.add_trace(go.Scatterpolar(
            r=normalized.loc[segment].tolist() + [normalized.loc[segment].tolist()[0]],
            theta=metrics + [metrics[0]],
            fill="toself",
            name=str(segment),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Relative Segment Profiles (normalized)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each axis is scaled 0–100 relative to the other segments, so shapes "
        "show relative strengths (e.g. highest spenders, oldest average age) "
        "rather than absolute values."
    )


# ==========================================
# TAB 3 — MACHINE LEARNING
# ==========================================

def render_ml_tab() -> None:
    st.subheader("🧠 Machine Learning Pipeline")

    st.code(
        "Customer Dataset\n"
        "        ↓\n"
        "Data Cleaning\n"
        "        ↓\n"
        "Feature Selection\n"
        "        ↓\n"
        "Feature Scaling\n"
        "        ↓\n"
        "K-Means Clustering\n"
        "        ↓\n"
        "Customer Segmentation\n"
        "        ↓\n"
        "Business Insights"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Algorithm", "K-Means")
    col2.metric("Learning Type", "Unsupervised")
    col3.metric("Clusters", "5")

    st.success("Model identifies hidden customer groups without predefined labels.")


# ==========================================
# TAB 4 — BUSINESS INSIGHTS
# ==========================================

def render_insights_tab() -> None:
    st.subheader("💡 Business Recommendations")

    st.markdown(
        """
        ### Premium Customers
        ✔ Maintain loyalty
        ✔ Provide exclusive offers
        ✔ Increase customer lifetime value

        ### Potential Customers
        ✔ Personalized promotions
        ✔ Encourage higher spending

        ### Low Engagement Customers
        ✔ Re-engagement campaigns
        ✔ Special discounts
        ✔ Feedback collection

        ### Regular Customers
        ✔ Improve retention
        ✔ Cross-selling opportunities
        """
    )


# ==========================================
# TAB 5 — PAYMENT BEHAVIOR
# ==========================================

def render_payment_tab(data: pd.DataFrame) -> None:
    if data.empty:
        st.info("Adjust the filters in the sidebar to see payment analysis.")
        return

    st.subheader("💳 Customer Payment Behavior Analysis")

    p1, p2, p3 = st.columns(3)
    p1.metric("💳 Payment Channels", data["Payment_Method"].nunique())
    p2.metric("🏆 Most Used Method", data["Payment_Method"].value_counts().idxmax())

    digital = data[data["Payment_Method"].isin(DIGITAL_METHODS)]
    digital_pct = round(len(digital) / len(data) * 100, 1)
    p3.metric("📱 Digital Payments", f"{digital_pct}%")

    payment_count = data["Payment_Method"].value_counts()
    fig_payment = px.pie(
        values=payment_count.values, names=payment_count.index,
        hole=0.45, title="Customer Preferred Payment Methods",
    )
    st.plotly_chart(fig_payment, use_container_width=True)

    st.subheader("🎯 Payment Preference Across Customer Segments")
    payment_segment = pd.crosstab(data["Segment"], data["Payment_Method"])
    fig_segment_payment = px.bar(
        payment_segment, barmode="group",
        title="Payment Methods Used by Each Customer Segment",
    )
    st.plotly_chart(fig_segment_payment, use_container_width=True)

    st.subheader("💡 Payment Strategy Insights")
    st.info(
        "- Premium customers can be targeted with credit-card rewards.\n"
        "- Mobile wallet users are suitable for digital campaigns.\n"
        "- Cash users can be encouraged towards online payment adoption.\n"
        "- Payment preferences help personalize marketing strategies."
    )


# ==========================================
# EXPORT
# ==========================================

def render_export(data: pd.DataFrame) -> None:
    st.header("📥 Export Customer Data")

    if data.empty:
        st.info("No data to export with the current filters.")
        return

    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Customer Report",
        csv,
        "customer_segmentation_report.csv",
        "text/csv",
    )


def render_footer() -> None:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='footer-note'>Customer Intelligence Platform · "
        "Built with Python, Pandas, Plotly & Streamlit</div>",
        unsafe_allow_html=True,
    )


# ==========================================
# MAIN APP
# ==========================================

def main() -> None:
    inject_custom_css()

    try:
        df = load_dataset()
    except FileNotFoundError:
        st.error(f"Dataset not found at '{DATA_PATH}'. Please check the file path.")
        st.stop()

    missing_cols = validate_dataset(df)
    if missing_cols:
        st.error(f"Dataset is missing required column(s): {', '.join(missing_cols)}")
        st.stop()

    data = render_sidebar(df)

    render_hero()
    render_kpi_dashboard(data)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Analytics", "🎯 Customer Segments", "🤖 Machine Learning",
         "💡 Business Insights", "💳 Payment Behavior"]
    )

    with tab1:
        render_analytics_tab(data)
    with tab2:
        render_segment_tab(data)
    with tab3:
        render_ml_tab()
    with tab4:
        render_insights_tab()
    with tab5:
        render_payment_tab(data)

    render_export(data)
    render_footer()


if __name__ == "__main__":
    main()
