import streamlit as st
import pandas as pd
import plotly.express as px
# Custom Styling

st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

h1 {
    color: #1f2937;
}

div[data-testid="metric-container"] {
    background-color: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="Customer Segmentation AI",
    page_icon="👥",
    layout="wide"
)

# Title
st.title("👥 Customer Segmentation AI Dashboard")
st.info(
"""
This AI-powered dashboard uses K-Means Clustering
to identify customer groups based on income and spending behavior.
"""
)
st.markdown(
    "### Analyze customer behavior using Machine Learning (K-Means Clustering)"
)

# Load data
df = pd.read_csv("final_customer_segments.csv")
# Sidebar Filter
st.sidebar.title("🔎 Customer Filter")

selected_segment = st.sidebar.multiselect(
    "Select Customer Segment",
    options=df["Segment"].unique(),
    default=df["Segment"].unique(),
    max_selections=5
)

filtered_df = df[
    df["Segment"].isin(selected_segment)
]

st.sidebar.metric(
    "Customers Selected",
    len(filtered_df)
)


# KPI Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Total Segments",
        filtered_df["Cluster"].nunique()
    )

with col3:
    st.metric(
        "Avg Spending Score",
        round(filtered_df["Spending Score (1-100)"].mean(),2)
    )


# Bar Chart
st.subheader("📊 Customer Segment Analysis")

segment_count = filtered_df["Segment"].value_counts()

fig = px.bar(
    x=segment_count.index,
    y=segment_count.values,
    title="Customers in Each Segment",
    labels={
        "x":"Segment",
        "y":"Number of Customers"
    }
)

st.plotly_chart(fig, use_container_width=True)


# Scatter Plot
st.subheader("🎯 Customer Segmentation Map")

fig2 = px.scatter(
    filtered_df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Segment",
    size="Age",
    hover_data=[
        "CustomerID",
        "Gender"
    ],
    title="Income vs Spending Behavior"
)

st.plotly_chart(fig2, use_container_width=True)


# Data Table
st.subheader("Customer Data")

st.dataframe(filtered_df)
# AI Customer Insights

st.subheader("🤖 AI Customer Insights")

segment_info = {
    "High Value Customers": 
    "These customers have high income and high spending behavior. Focus on premium offers, loyalty programs, and exclusive deals.",
    
    "Potential Customers":
    "These customers show moderate spending patterns. Use targeted promotions to increase engagement.",
    
    "Low Engagement Customers":
    "These customers have lower spending activity. Use discounts and personalized campaigns to improve retention."
}

for segment in filtered_df["Segment"].unique():
    with st.expander(f"💡 {segment}"):
        if segment in segment_info:
            st.write(segment_info[segment])
        else:
            st.write("Analyze this segment based on income and spending behavior.")
