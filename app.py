import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ==========================
# PAGE CONFIGURATION
# ==========================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)


# ==========================
# CUSTOM DESIGN
# ==========================

st.markdown("""
<style>

body {
    background-color:#f5f7fb;
}

.main {
    background-color:#f5f7fb;
}


[data-testid="metric-container"] {

background:white;
padding:20px;
border-radius:18px;
box-shadow:0 4px 12px rgba(0,0,0,0.08);

}


h1 {

color:#0f172a;

}


h2 {

color:#1e293b;

}


.card {

background:white;
padding:20px;
border-radius:15px;
box-shadow:0 3px 10px rgba(0,0,0,0.08);

}


</style>

""", unsafe_allow_html=True)



# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "final_customer_segments.csv"
)



# ==========================
# HEADER
# ==========================

st.title(
    "📊 Customer Intelligence & Segmentation Platform"
)


st.markdown(
"""
### AI-powered customer analytics system

Discover customer behavior patterns using
**Machine Learning + K-Means Clustering**
"""
)


st.info(
"""
This platform analyzes customer income,
spending behavior, and demographic patterns
to generate actionable business insights.
"""
)



# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("🔎 Customer Analytics")


segments = df["Segment"].unique()


selected = st.sidebar.multiselect(
    "Select Segment",
    segments,
    default=segments
)


filtered_df = df[
    df["Segment"].isin(selected)
]


st.sidebar.metric(
    "Customers Selected",
    len(filtered_df)
)
# ==========================
# EXECUTIVE OVERVIEW
# ==========================

st.header("📌 Executive Overview")


col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "👥 Total Customers",
        len(filtered_df)
    )


with col2:
    st.metric(
        "🎯 Customer Segments",
        filtered_df["Segment"].nunique()
    )


with col3:
    st.metric(
        "💰 Average Income",
        f"{filtered_df['Annual Income (k$)'].mean():.2f}K"
    )


with col4:
    st.metric(
        "🛒 Average Spending",
        f"{filtered_df['Spending Score (1-100)'].mean():.2f}"
    )



# ==========================
# SEGMENT DISTRIBUTION
# ==========================

st.header("📊 Customer Segment Distribution")


segment_count = filtered_df["Segment"].value_counts()


fig1 = px.pie(
    values=segment_count.values,
    names=segment_count.index,
    hole=0.45,
    title="Customer Distribution by Segment"
)


st.plotly_chart(
    fig1,
    use_container_width=True
)



# ==========================
# CUSTOMER BEHAVIOR MAP
# ==========================

st.header("🎯 Customer Behavior Intelligence")


fig2 = px.scatter(
    filtered_df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Segment",
    size="Age",
    hover_data=[
        "CustomerID",
        "Gender",
        "Age"
    ],
    title="Income vs Spending Relationship"
)


st.plotly_chart(
    fig2,
    use_container_width=True
)



# ==========================
# AGE ANALYSIS
# ==========================

st.header("👥 Customer Age Analysis")


fig3 = px.histogram(
    filtered_df,
    x="Age",
    nbins=15,
    title="Age Distribution"
)


st.plotly_chart(
    fig3,
    use_container_width=True
)



# ==========================
# INCOME ANALYSIS
# ==========================

st.header("💰 Income Distribution")


fig4 = px.histogram(
    filtered_df,
    x="Annual Income (k$)",
    nbins=15,
    title="Annual Income Distribution"
)


st.plotly_chart(
    fig4,
    use_container_width=True
)
