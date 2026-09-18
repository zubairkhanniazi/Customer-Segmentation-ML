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
