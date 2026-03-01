import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Model loading
from src.phase7_deployment.model_loader import load_model
from src.phase3_model.predict import predict_probabilities

# Scoring
from src.phase5_scoring.credit_score import (
    generate_credit_scores,
    assign_risk_category,
    assign_fraud_label
)

# Feature engineering
from src.phase2_features.feature_engineering import engineer_features


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡",
    layout="wide"
)

st.title("🛡 Financial Fraud Detection Dashboard")
st.write("Upload transactions to analyze fraud risk")


# =========================
# LOAD MODEL
# =========================

model, feature_names = load_model()


# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success(f"{len(df)} transactions loaded")

    # =========================
    # FEATURE ENGINEERING
    # =========================

    X = engineer_features(df)

    # Align features
    X = X[feature_names]


    # =========================
    # PREDICT
    # =========================

    probabilities = predict_probabilities(model, X)

    scores = generate_credit_scores(probabilities)

    risk_categories = assign_risk_category(scores)

    fraud_labels = assign_fraud_label(probabilities)


    # =========================
    # CREATE RESULTS DF
    # =========================

    results = df.copy()

    results["Fraud Probability"] = probabilities
    results["Safety Score"] = scores
    results["Risk Category"] = risk_categories
    results["Fraud Label"] = fraud_labels


    # =========================
    # SUMMARY METRICS
    # =========================

    st.header("📊 Summary")

    col1, col2, col3, col4 = st.columns(4)

    total = len(results)
    high_risk = (results["Risk Category"] == "High Risk").sum()
    medium_risk = (results["Risk Category"] == "Fair").sum()
    fraud_count = (results["Fraud Label"] == "Fraud").sum()

    col1.metric("Total Transactions", total)
    col2.metric("High Risk", high_risk)
    col3.metric("Medium Risk", medium_risk)
    col4.metric("Fraud Detected", fraud_count)


    # =========================
    # RISK DISTRIBUTION CHART
    # =========================

    st.header("📈 Risk Distribution")

    fig = px.pie(
        results,
        names="Risk Category",
        title="Risk Category Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)


    # =========================
    # PROBABILITY DISTRIBUTION
    # =========================

    st.header("📉 Fraud Probability Distribution")

    fig2 = px.histogram(
        results,
        x="Fraud Probability",
        nbins=50
    )

    st.plotly_chart(fig2, use_container_width=True)


    # =========================
    # RESULTS TABLE
    # =========================

    st.header("📋 Transaction Results")

    st.dataframe(
        results.sort_values(
            "Fraud Probability",
            ascending=False
        ),
        use_container_width=True
    )


    # =========================
    # DOWNLOAD BUTTON
    # =========================

    csv = results.to_csv(index=False)

    st.download_button(
        label="Download Results CSV",
        data=csv,
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )


    # =========================
    # INDIVIDUAL TRANSACTION VIEW
    # =========================

    st.header("🔍 Transaction Detail")

    selected_index = st.number_input(
        "Select Transaction Index",
        min_value=0,
        max_value=len(results)-1,
        value=0
    )

    selected_row = results.iloc[selected_index]

    st.write(selected_row)