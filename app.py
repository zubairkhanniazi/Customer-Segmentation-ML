import streamlit as st
import pandas as pd
import plotly.express as px


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)


# ==========================
# STYLE
# ==========================

st.markdown("""
<style>

.main{
background-color:#f5f7fb;
}

h1{
color:#0f172a;
}

h2{
color:#1e293b;
}


div[data-testid="metric-container"]{

background:white;
padding:18px;
border-radius:15px;
box-shadow:0px 4px 12px rgba(0,0,0,0.08);

}


</style>

""",
unsafe_allow_html=True)



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
### AI Powered Customer Analytics System

Machine Learning based customer behavior analysis
using **K-Means Clustering**
"""
)


st.info(
"""
This platform discovers hidden customer groups
and converts them into actionable business insights.
"""
)



# ==========================
# SIDEBAR
# ==========================

st.sidebar.title(
"🔎 Customer Filter"
)


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
"Customers Selected",
len(filtered_df)
)



# ==========================
# EXECUTIVE DASHBOARD
# ==========================

st.header(
"📌 Executive Overview"
)


c1,c2,c3,c4 = st.columns(4)


with c1:

    st.metric(
    "👥 Total Customers",
    len(filtered_df)
    )


with c2:

    st.metric(
    "🎯 Segments",
    filtered_df["Segment"].nunique()
    )


with c3:

    st.metric(
    "💰 Avg Income",
    f"{filtered_df['Annual Income (k$)'].mean():.1f}K"
    )


with c4:

    st.metric(
    "🛒 Avg Spending",
    f"{filtered_df['Spending Score (1-100)'].mean():.1f}"
    )



# ==========================
# SEGMENT DISTRIBUTION
# ==========================


st.header(
"📈 Customer Segment Distribution"
)


segment_count = filtered_df["Segment"].value_counts()


fig1 = px.pie(

values=segment_count.values,

names=segment_count.index,

hole=0.45,

title="Customer Segment Share"

)


st.plotly_chart(
fig1,
use_container_width=True
)



# ==========================
# CUSTOMER BEHAVIOR
# ==========================


st.header(
"🎯 Customer Behavior Analysis"
)


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


st.header(
"👥 Customer Age Analysis"
)


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
# SEGMENT INTELLIGENCE
# ==========================


st.header(
"🤖 Customer Segment Intelligence"
)


for segment in filtered_df["Segment"].unique():


    data = filtered_df[
    filtered_df["Segment"]==segment
    ]


    with st.expander(
    f"⭐ {segment}"
    ):


        a,b,c = st.columns(3)


        a.metric(
        "Customers",
        len(data)
        )


        b.metric(
        "Average Income",
        round(
        data["Annual Income (k$)"].mean(),
        2
        )
        )


        c.metric(
        "Average Spending",
        round(
        data["Spending Score (1-100)"].mean(),
        2
        )
        )


        st.write(
        """
        Recommended Strategy:

        ✅ Personalized marketing

        ✅ Loyalty programs

        ✅ Targeted promotions

        ✅ Customer retention campaigns

        """
        )



# ==========================
# MACHINE LEARNING EXPLANATION
# ==========================


st.header(
"🧠 Machine Learning Methodology"
)


st.markdown(
"""

### Algorithm:
**K-Means Clustering**

### Learning Type:
Unsupervised Machine Learning


### Workflow:


Customer Dataset
↓
Data Cleaning
↓
Feature Selection
↓
Feature Scaling
↓
K-Means Algorithm
↓
Customer Segmentation
↓
Business Insights



"""
)



# ==========================
# DOWNLOAD REPORT
# ==========================


st.header(
"📥 Export Report"
)


csv = filtered_df.to_csv(
index=False
)


st.download_button(

"Download Customer Report",

csv,

"customer_segments_report.csv",

"text/csv"

)



# ==========================
# FOOTER
# ==========================


st.markdown(
"""
---
🚀 Built with Python | Scikit-Learn | K-Means | Streamlit
"""
)
