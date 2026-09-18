import streamlit as st
import pandas as pd
import plotly.express as px


# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)


# ----------------------------
# Custom CSS
# ----------------------------

st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

h1{
    color:#1f2937;
}

[data-testid="metric-container"]{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)



# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv(
    "final_customer_segments.csv"
)



# ----------------------------
# Header
# ----------------------------

st.title(
    "📊 Customer Intelligence & Segmentation Platform"
)

st.markdown(
"""
Machine Learning powered customer behavior analysis using **K-Means Clustering**
"""
)


st.info(
"""
This platform identifies hidden customer groups based on income,
spending behavior, and demographic patterns.
"""
)



# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("🔎 Customer Filter")


segments = df["Segment"].unique()


selected_segment = st.sidebar.multiselect(
    "Select Customer Segment",
    segments,
    default=segments
)


filtered_df = df[
    df["Segment"].isin(selected_segment)
]


st.sidebar.metric(
    "Selected Customers",
    len(filtered_df)
)



# ----------------------------
# Executive KPIs
# ----------------------------

st.header("📌 Executive Overview")


col1,col2,col3,col4 = st.columns(4)


with col1:
    st.metric(
        "Total Customers",
        len(filtered_df)
    )


with col2:
    st.metric(
        "Customer Segments",
        filtered_df["Segment"].nunique()
    )


with col3:
    st.metric(
        "Average Income",
        round(filtered_df["Annual Income (k$)"].mean(),2)
    )


with col4:
    st.metric(
        "Average Spending",
        round(filtered_df["Spending Score (1-100)"].mean(),2)
    )



# ----------------------------
# Segment Distribution
# ----------------------------

st.header("📈 Customer Segment Analysis")


segment_count = filtered_df["Segment"].value_counts()


fig = px.pie(
    values=segment_count.values,
    names=segment_count.index,
    title="Customer Distribution by Segment",
    hole=0.4
)


st.plotly_chart(
    fig,
    use_container_width=True
)



# ----------------------------
# Customer Behavior Map
# ----------------------------

st.header(
    "🎯 Income vs Spending Behavior"
)


fig2 = px.scatter(
    filtered_df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Segment",
    size="Age",
    hover_data=[
        "CustomerID",
        "Gender"
    ]
)


st.plotly_chart(
    fig2,
    use_container_width=True
)



# ----------------------------
# Segment Profile
# ----------------------------

st.header(
    "🤖 Customer Intelligence"
)


for segment in filtered_df["Segment"].unique():

    data = filtered_df[
        filtered_df["Segment"]==segment
    ]

    with st.expander(
        f"⭐ {segment}"
    ):

        col1,col2,col3 = st.columns(3)


        col1.metric(
            "Customers",
            len(data)
        )


        col2.metric(
            "Avg Income",
            round(data["Annual Income (k$)"].mean(),2)
        )


        col3.metric(
            "Avg Spending",
            round(data["Spending Score (1-100)"].mean(),2)
        )


        st.write(
        """
        Business Strategy:

        • Personalized marketing campaigns  
        • Customer retention programs  
        • Targeted offers based on behavior
        """
        )



# ----------------------------
# Download Report
# ----------------------------

st.header(
    "📥 Export Customer Report"
)


csv = filtered_df.to_csv(
    index=False
)


st.download_button(
    "Download Segmentation Report",
    csv,
    "customer_segments_report.csv",
    "text/csv"
)



# ----------------------------
# Footer
# ----------------------------

st.markdown(
"""
---
🚀 Built with Python | Scikit-Learn | K-Means Clustering | Streamlit
"""
)
