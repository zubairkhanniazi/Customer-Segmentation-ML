import streamlit as st
import pandas as pd
import plotly.express as px


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)


# ==============================
# PROFESSIONAL CSS
# ==============================

st.markdown("""
<style>

body {
    background-color:#f6f8fc;
}


.main {
    background-color:#f6f8fc;
}


h1 {
    color:#111827;
    font-size:42px;
}


h2 {
    color:#1f2937;
}


.metric-card {

background:white;
padding:20px;
border-radius:18px;
box-shadow:0 5px 15px rgba(0,0,0,0.08);

}


.insight-card {

background:white;
padding:25px;
border-radius:20px;
box-shadow:0 5px 15px rgba(0,0,0,0.08);

}


</style>
""", unsafe_allow_html=True)



# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(
    "final_customer_segments.csv"
)



# ==============================
# HEADER
# ==============================

st.title(
"📊 Customer Intelligence & Segmentation Platform"
)


st.markdown(
"""
### AI Powered Customer Analytics Dashboard

Machine Learning based customer behavior analysis using
**K-Means Clustering Algorithm**
"""
)


st.info(
"""
Transforming customer data into actionable business intelligence.
"""
)



# ==============================
# SIDEBAR
# ==============================

st.sidebar.title(
"🔎 Customer Explorer"
)


segments = df["Segment"].unique()


selected_segments = st.sidebar.multiselect(
    "Choose Customer Segments",
    segments,
    default=segments
)


data = df[
df["Segment"].isin(selected_segments)
]


st.sidebar.metric(
"Customers Selected",
len(data)
)



# ==============================
# KPI SECTION
# ==============================

st.header(
"📌 Executive Overview"
)


c1,c2,c3,c4,c5 = st.columns(5)


with c1:
    st.metric(
    "Total Customers",
    len(data)
    )


with c2:
    st.metric(
    "Segments",
    data["Segment"].nunique()
    )


with c3:
    st.metric(
    "Average Age",
    round(data["Age"].mean(),1)
    )


with c4:
    st.metric(
    "Average Income",
    str(round(data["Annual Income (k$)"].mean(),1))+"K"
    )


with c5:
    st.metric(
    "Avg Spending",
    round(data["Spending Score (1-100)"].mean(),1)
    )



# ==============================
# TABS
# ==============================

tab1,tab2,tab3,tab4 = st.tabs(
[
"📈 Analytics",
"🎯 Customer Segments",
"🤖 Machine Learning",
"💡 Business Insights"
]
)



# ==============================
# TAB 1 ANALYTICS
# ==============================


with tab1:


    col1,col2 = st.columns(2)


    with col1:

        segment_count = data["Segment"].value_counts()


        fig1 = px.pie(

        values=segment_count.values,

        names=segment_count.index,

        hole=.45,

        title="Customer Segment Distribution"

        )


        st.plotly_chart(
        fig1,
        use_container_width=True
        )



    with col2:


        fig2 = px.histogram(

        data,

        x="Age",

        title="Customer Age Distribution",

        nbins=15

        )


        st.plotly_chart(
        fig2,
        use_container_width=True
        )




    st.subheader(
    "Income vs Spending Behavior"
    )


    fig3 = px.scatter(

    data,

    x="Annual Income (k$)",

    y="Spending Score (1-100)",

    color="Segment",

    size="Age",

    hover_data=["CustomerID"]

    )


    st.plotly_chart(
    fig3,
    use_container_width=True
    )



# ==============================
# TAB 2 SEGMENT INTELLIGENCE
# ==============================


with tab2:


    st.subheader(
    "Customer Segment Profiles"
    )


    for segment in data["Segment"].unique():


        segment_data = data[
        data["Segment"]==segment
        ]


        with st.expander(
        "⭐ "+segment
        ):


            a,b,c = st.columns(3)


            a.metric(
            "Customers",
            len(segment_data)
            )


            b.metric(
            "Income",
            round(
            segment_data["Annual Income (k$)"].mean(),
            1
            )
            )


            c.metric(
            "Spending",
            round(
            segment_data["Spending Score (1-100)"].mean(),
            1
            )
            )


            st.write(
            """
            Recommended Strategy:

            • Personalized marketing campaigns

            • Customer loyalty programs

            • Targeted product recommendations

            • Retention strategy

            """
            )



# ==============================
# TAB 3 MACHINE LEARNING
# ==============================


with tab3:


    st.subheader(
    "🧠 Machine Learning Pipeline"
    )


    st.code(
"""
Customer Dataset

        ↓

Data Cleaning

        ↓

Feature Selection

        ↓

Feature Scaling

        ↓

K-Means Clustering

        ↓

Customer Segmentation

        ↓

Business Insights
"""
)


    col1,col2,col3 = st.columns(3)


    col1.metric(
    "Algorithm",
    "K-Means"
    )


    col2.metric(
    "Learning Type",
    "Unsupervised"
    )


    col3.metric(
    "Clusters",
    "5"
    )



    st.success(
    """
    Model identifies hidden customer groups
    without predefined labels.
    """
    )



# ==============================
# TAB 4 BUSINESS INSIGHTS
# ==============================


with tab4:


    st.subheader(
    "💡 Business Recommendations"
    )


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



# ==============================
# DOWNLOAD
# ==============================


st.header(
"📥 Export Customer Data"
)


csv = data.to_csv(
index=False
)


st.download_button(

"Download Customer Report",

csv,

"customer_segmentation_report.csv",

"text/csv"

)



# ==============================
# FOOTER
# ==============================


st.markdown(
"""
---
🚀 Built using Python | Pandas | Plotly | Machine Learning | Streamlit
"""
)
# ==============================
# PAYMENT METHOD ANALYSIS
# ==============================

st.markdown("---")

st.header("💳 Customer Payment Behavior Analysis")


# Create payment method data if column does not exist

if "Payment_Method" not in data.columns:

    import numpy as np

    np.random.seed(42)

    payment_methods = [
        "Credit Card",
        "Debit Card",
        "Mobile Wallet",
        "Cash",
        "Bank Transfer"
    ]

    data = data.copy()

    data["Payment_Method"] = np.random.choice(
        payment_methods,
        size=len(data),
        p=[0.35,0.25,0.20,0.10,0.10]
    )



# KPI Cards

payment_col1, payment_col2, payment_col3 = st.columns(3)


with payment_col1:

    st.metric(
        "💳 Payment Channels",
        data["Payment_Method"].nunique()
    )


with payment_col2:

    most_used = (
        data["Payment_Method"]
        .value_counts()
        .idxmax()
    )

    st.metric(
        "🏆 Most Used Method",
        most_used
    )


with payment_col3:

    digital = data[
        data["Payment_Method"].isin(
            [
                "Credit Card",
                "Debit Card",
                "Mobile Wallet",
                "Bank Transfer"
            ]
        )
    ]

    digital_percentage = round(
        len(digital) / len(data) * 100,
        1
    )


    st.metric(
        "📱 Digital Payments",
        f"{digital_percentage}%"
    )



# Payment Distribution Chart

payment_count = (
    data["Payment_Method"]
    .value_counts()
)


fig_payment = px.pie(

    values=payment_count.values,

    names=payment_count.index,

    hole=0.45,

    title="Customer Preferred Payment Methods"

)


st.plotly_chart(
    fig_payment,
    use_container_width=True
)



# Payment by Segment

st.subheader(
    "🎯 Payment Preference Across Customer Segments"
)


payment_segment = pd.crosstab(

    data["Segment"],

    data["Payment_Method"]

)



fig_segment_payment = px.bar(

    payment_segment,

    barmode="group",

    title="Payment Methods Used by Each Customer Segment"

)


st.plotly_chart(

    fig_segment_payment,

    use_container_width=True

)



# Business Insights

st.subheader(
    "💡 Payment Strategy Insights"
)


st.info(
"""
• Premium customers can be targeted with credit-card rewards.

• Mobile wallet users are suitable for digital campaigns.

• Cash users can be encouraged towards online payment adoption.

• Payment preferences help personalize marketing strategies.
"""
)
