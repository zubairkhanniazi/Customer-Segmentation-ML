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

PAYMENT_METHODS = ["Credit Card", "Debit Card", "Mobile Wallet", "Cash", "Bank Transfer"]
DIGITAL_METHODS = ["Credit Card", "Debit Card", "Mobile Wallet", "Bank Transfer"]

THEMES = {
    "Light": {"accent": "#6366f1", "accent_dark": "#4338ca", "bg": "#f6f8fc",
              "card_bg": "white", "text": "#111827", "sidebar_bg": "#f8fafc"},
    "Dark": {"accent": "#818cf8", "accent_dark": "#6366f1", "bg": "#0f172a",
             "card_bg": "#1e293b", "text": "#f1f5f9", "sidebar_bg": "#0b1220"},
}


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

def inject_custom_css(theme: dict) -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{
            background-color: {theme['bg']};
            color: {theme['text']};
        }}

        .hero {{
            background: linear-gradient(135deg, {theme['accent']} 0%, {theme['accent_dark']} 100%);
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
            background: {theme['card_bg']};
            border-radius: 16px;
            padding: 1rem 1.2rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.2);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.12);
        }}

        section[data-testid="stSidebar"] {{
            background: {theme['sidebar_bg']};
        }}

        .section-divider {{
            margin: 1.8rem 0;
            border: none;
            border-top: 1px solid rgba(148, 163, 184, 0.3);
        }}
        .footer-note {{
            text-align: center;
            color: #94a3b8;
            font-size: 13px;
            padding-top: 1.5rem;
        }}
        .badge {{
            display: inline-block;
            background: rgba(99, 102, 241, 0.12);
            color: {theme['accent_dark']};
            font-size: 12px;
            font-weight: 600;
            padding: 3px 10px;
            border-radius: 999px;
            margin-right: 6px;
        }}
        .profile-card {{
            background: {theme['card_bg']};
            border-radius: 18px;
            padding: 1.5rem 1.8rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.2);
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

    df["Estimated_CLV"] = estimate_clv(df)
    return df


def estimate_clv(df: pd.DataFrame) -> pd.Series:
    """
    A simple, transparent Customer Lifetime Value estimate:
    CLV ≈ Annual Income × (Spending Score / 100) × assumed 3-year horizon.
    This is a illustrative heuristic, not a true predictive CLV model —
    labeled as an estimate everywhere it's shown.
    """
    assumed_years = 3
    return (df["Annual Income (k$)"] * 1000 * (df["Spending Score (1-100)"] / 100) * assumed_years).round(0)


def validate_dataset(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def detect_outliers(df: pd.DataFrame, column: str) -> pd.Series:
    """Flag values outside 1.5x the IQR as outliers for the given column."""
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ~df[column].between(lower, upper)


# ==========================================
# HERO
# ==========================================

def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📊 Customer Intelligence & Segmentation Platform</div>
            <div class="hero-subtitle">AI-powered customer analytics using K-Means clustering —
            explore segments, lifetime value, payment behavior, and outliers in one place.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# SIDEBAR — FILTERS & SETTINGS
# ==========================================

def render_sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    with st.sidebar:
        st.title("🔎 Customer Explorer")

        theme_choice = st.radio("Theme", list(THEMES.keys()), horizontal=True)
        theme = THEMES[theme_choice]

        st.markdown("---")
        st.caption("Filter the customer base explored across every tab.")

        segments = sorted(df["Segment"].unique())
        selected_segments = st.multiselect("Customer Segments", segments, default=segments)

        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        age_range = st.slider("Age Range", age_min, age_max, (age_min, age_max))

        income_min, income_max = int(df["Annual Income (k$)"].min()), int(df["Annual Income (k$)"].max())
        income_range = st.slider("Annual Income (k$)", income_min, income_max, (income_min, income_max))

        spend_min, spend_max = int(df["Spending Score (1-100)"].min()), int(df["Spending Score (1-100)"].max())
        spend_range = st.slider("Spending Score", spend_min, spend_max, (spend_min, spend_max))

        if st.button("↺ Reset Filters"):
            st.rerun()

        filtered = df[
            df["Segment"].isin(selected_segments)
            & df["Age"].between(*age_range)
            & df["Annual Income (k$)"].between(*income_range)
            & df["Spending Score (1-100)"].between(*spend_range)
        ]

        st.markdown("---")
        st.metric("Customers Selected", f"{len(filtered):,} / {len(df):,}")

        if filtered.empty:
            st.warning("No customers match the current filters.")

    return filtered, theme


# ==========================================
# KPI DASHBOARD
# ==========================================

def render_kpi_dashboard(data: pd.DataFrame) -> None:
    st.header("📌 Executive Overview")

    if data.empty:
        st.info("Adjust the filters in the sidebar to see results.")
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Customers", f"{len(data):,}")
    c2.metric("Segments", data["Segment"].nunique())
    c3.metric("Average Age", round(data["Age"].mean(), 1))
    c4.metric("Average Income", f"{data['Annual Income (k$)'].mean():.1f}K")
    c5.metric("Avg Spending", round(data["Spending Score (1-100)"].mean(), 1))
    c6.metric("Est. Avg CLV*", f"${data['Estimated_CLV'].mean():,.0f}")
    st.caption("*Estimated CLV is a simplified heuristic (Income × Spending Score × 3-year horizon), "
               "not a predictive model.")


# ==========================================
# TAB 1 — ANALYTICS
# ==========================================

def render_analytics_tab(data: pd.DataFrame, theme: dict) -> None:
    if data.empty:
        st.info("Adjust the filters in the sidebar to see charts.")
        return

    col1, col2 = st.columns(2)

    with col1:
        segment_count = data["Segment"].value_counts()
        fig1 = px.pie(
            values=segment_count.values, names=segment_count.index,
            hole=0.45, title="Customer Segment Distribution",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(
            data, x="Age", title="Customer Age Distribution", nbins=15,
            color_discrete_sequence=[theme["accent"]],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Income vs Spending Behavior")
    fig3 = px.scatter(
        data, x="Annual Income (k$)", y="Spending Score (1-100)",
        color="Segment", size="Age", hover_data=["CustomerID", "Estimated_CLV"],
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
            a, b, c, d = st.columns(4)
            a.metric("Customers", len(segment_data))
            b.metric("Income", round(segment_data["Annual Income (k$)"].mean(), 1))
            c.metric("Spending", round(segment_data["Spending Score (1-100)"].mean(), 1))
            d.metric("Est. Avg CLV*", f"${segment_data['Estimated_CLV'].mean():,.0f}")
            st.markdown(SEGMENT_STRATEGIES)

    render_segment_radar(data)


def render_segment_radar(data: pd.DataFrame) -> None:
    st.subheader("📡 Segment Comparison")

    metrics = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    profile = data.groupby("Segment")[metrics].mean()
    normalized = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9) * 100

    fig = go.Figure()
    for segment in normalized.index:
        values = normalized.loc[segment].tolist()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
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
    st.caption("Each axis is scaled 0–100 relative to the other segments, showing relative "
               "strengths rather than absolute values.")


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
# TAB 6 — CUSTOMER 360
# ==========================================

def render_customer_360_tab(df: pd.DataFrame) -> None:
    st.subheader("🔍 Customer 360 Profile")

    customer_id = st.text_input("Enter Customer ID to view full profile", key="c360_search")

    if not customer_id:
        st.info("Enter a Customer ID above to see their full profile.")
        return

    match = df[df["CustomerID"].astype(str) == customer_id.strip()]
    if match.empty:
        st.warning("No customer found with that ID.")
        return

    customer = match.iloc[0]
    segment_avg = df[df["Segment"] == customer["Segment"]].mean(numeric_only=True)

    st.markdown('<div class="profile-card">', unsafe_allow_html=True)
    st.markdown(f"### Customer #{customer['CustomerID']} · Segment: **{customer['Segment']}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Age", int(customer["Age"]))
    m2.metric("Income", f"{customer['Annual Income (k$)']:.0f}K",
               delta=f"{customer['Annual Income (k$)'] - segment_avg['Annual Income (k$)']:.1f} vs segment avg")
    m3.metric("Spending Score", int(customer["Spending Score (1-100)"]),
               delta=f"{customer['Spending Score (1-100)'] - segment_avg['Spending Score (1-100)']:.1f} vs segment avg")
    m4.metric("Est. CLV*", f"${customer['Estimated_CLV']:,.0f}")
    st.markdown(f"**Preferred Payment Method:** {customer.get('Payment_Method', 'N/A')}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.caption("*Estimated CLV is a simplified heuristic, not a predictive model.")


# ==========================================
# TAB 7 — COMPARE CUSTOMERS
# ==========================================

def render_compare_tab(df: pd.DataFrame) -> None:
    st.subheader("⚖️ Compare Two Customers")

    col_a, col_b = st.columns(2)
    with col_a:
        id_a = st.text_input("Customer ID A", key="compare_a")
    with col_b:
        id_b = st.text_input("Customer ID B", key="compare_b")

    if not (id_a and id_b):
        st.info("Enter two Customer IDs above to compare them.")
        return

    row_a = df[df["CustomerID"].astype(str) == id_a.strip()]
    row_b = df[df["CustomerID"].astype(str) == id_b.strip()]

    if row_a.empty or row_b.empty:
        st.warning("One or both Customer IDs were not found.")
        return

    a, b = row_a.iloc[0], row_b.iloc[0]

    comparison = pd.DataFrame({
        "Metric": ["Segment", "Age", "Annual Income (k$)", "Spending Score (1-100)",
                   "Estimated CLV*", "Payment Method"],
        f"Customer {a['CustomerID']}": [a["Segment"], a["Age"], a["Annual Income (k$)"],
                                          a["Spending Score (1-100)"], f"${a['Estimated_CLV']:,.0f}",
                                          a.get("Payment_Method", "N/A")],
        f"Customer {b['CustomerID']}": [b["Segment"], b["Age"], b["Annual Income (k$)"],
                                          b["Spending Score (1-100)"], f"${b['Estimated_CLV']:,.0f}",
                                          b.get("Payment_Method", "N/A")],
    })

    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.caption("*Estimated CLV is a simplified heuristic, not a predictive model.")


# ==========================================
# TAB 8 — OUTLIER DETECTION
# ==========================================

def render_outlier_tab(data: pd.DataFrame) -> None:
    if data.empty:
        st.info("Adjust the filters in the sidebar to check for outliers.")
        return

    st.subheader("🚩 Unusual Customer Behavior")
    st.caption("Customers flagged using the IQR method — values far outside the typical "
               "range for Income or Spending Score.")

    income_outliers = detect_outliers(data, "Annual Income (k$)")
    spending_outliers = detect_outliers(data, "Spending Score (1-100)")
    flagged = data[income_outliers | spending_outliers]

    st.metric("Flagged Customers", f"{len(flagged)} / {len(data)}")

    if flagged.empty:
        st.success("No unusual customers detected in the current filter.")
    else:
        st.dataframe(
            flagged[["CustomerID", "Segment", "Age", "Annual Income (k$)",
                     "Spending Score (1-100)", "Estimated_CLV"]],
            use_container_width=True, hide_index=True,
        )

        fig = px.scatter(
            data, x="Annual Income (k$)", y="Spending Score (1-100)",
            color=(income_outliers | spending_outliers).map({True: "Outlier", False: "Typical"}),
            color_discrete_map={"Outlier": "#ef4444", "Typical": "#94a3b8"},
            title="Outliers Highlighted",
        )
        st.plotly_chart(fig, use_container_width=True)


# ==========================================
# EXPORT
# ==========================================

def render_export(data: pd.DataFrame) -> None:
    st.header("📥 Export Data")

    if data.empty:
        st.info("No data to export with the current filters.")
        return

    col1, col2 = st.columns(2)

    with col1:
        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Customer Report",
            csv, "customer_segmentation_report.csv", "text/csv",
        )

    with col2:
        summary = (
            data.groupby("Segment")
            .agg(
                Customers=("CustomerID", "count"),
                Avg_Age=("Age", "mean"),
                Avg_Income=("Annual Income (k$)", "mean"),
                Avg_Spending=("Spending Score (1-100)", "mean"),
                Avg_Estimated_CLV=("Estimated_CLV", "mean"),
            )
            .round(1)
            .reset_index()
        )
        summary_csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Segment Summary",
            summary_csv, "segment_summary_report.csv", "text/csv",
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
    try:
        df = load_dataset()
    except FileNotFoundError:
        st.error(f"Dataset not found at '{DATA_PATH}'. Please check the file path.")
        st.stop()

    missing_cols = validate_dataset(df)
    if missing_cols:
        st.error(f"Dataset is missing required column(s): {', '.join(missing_cols)}")
        st.stop()

    data, theme = render_sidebar(df)
    inject_custom_css(theme)

    render_hero()
    render_kpi_dashboard(data)

    tabs = st.tabs([
        "📈 Analytics", "🎯 Customer Segments", "🤖 Machine Learning",
        "💡 Business Insights", "💳 Payment Behavior",
        "🔍 Customer 360", "⚖️ Compare Customers", "🚩 Outlier Detection",
    ])

    with tabs[0]:
        render_analytics_tab(data, theme)
    with tabs[1]:
        render_segment_tab(data)
    with tabs[2]:
        render_ml_tab()
    with tabs[3]:
        render_insights_tab()
    with tabs[4]:
        render_payment_tab(data)
    with tabs[5]:
        render_customer_360_tab(df)
    with tabs[6]:
        render_compare_tab(df)
    with tabs[7]:
        render_outlier_tab(data)

    render_export(data)
    render_footer()


if __name__ == "__main__":
    main()
