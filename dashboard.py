import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
st.set_page_config(page_title="Credit Risk Analysis Dashboard", layout="wide")
script_dir = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "RandomForest": os.path.join(script_dir, "randomforest.pkl"),
    "GradientBoosting": os.path.join(script_dir, "gradientboosting.pkl"),
    "XGBoost": os.path.join(script_dir, "xgboost.pkl")
}

models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

# Load dataset for initial visualization
dataset_path = os.path.join(script_dir, "realistic_credit_risk_data.csv")
df = pd.read_csv(dataset_path)

# Initialize session state for prediction history and scatter plot data
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(columns=["credit_score", "predicted_risk_score"])

if "scatter_data" not in st.session_state:
    st.session_state.scatter_data = df[["credit_score", "risk_score"]].copy()

# Sidebar User Inputs
st.sidebar.header("Enter Credit Details")
credit_score = st.sidebar.number_input("Credit Score (300-850)", min_value=300, max_value=850, value=600)
credit_limit = st.sidebar.number_input("Credit Limit ($)", min_value=1000, max_value=100000, value=10000)
total_overdue_payments = st.sidebar.number_input("Total Overdue Payments", min_value=0, max_value=100, value=0)
highest_balance = st.sidebar.number_input("Highest Balance Ever ($)", min_value=1000, max_value=100000, value=5000)
current_balance = st.sidebar.number_input("Current Balance ($)", min_value=0, max_value=100000, value=2000)
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=50000)
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
employment_status = st.sidebar.selectbox("Employment Status", ["Unemployed", "Part-Time", "Full-Time"])
duration_of_credit = st.sidebar.number_input("Duration of Credit (Years)", min_value=1, max_value=30, value=5)

employment_status_map = {"Unemployed": 0, "Part-Time": 1, "Full-Time": 2}

# Predict Button
if st.sidebar.button("Predict Risk"):
    input_data = {
        "credit_score": credit_score,
        "credit_limit": credit_limit,
        "total_overdue_payments": total_overdue_payments,
        "highest_balance": highest_balance,
        "current_balance": current_balance,
        "income": income,
        "age": age,
        "employment_status": employment_status_map[employment_status],
        "duration_of_credit": duration_of_credit,
    }
    
    input_df = pd.DataFrame([input_data])
    risk_scores = {name: model.predict(input_df)[0] for name, model in models.items()}
    average_risk_score = np.mean(list(risk_scores.values()))
    
    # Update prediction history and scatter data
    st.session_state.prediction_history = pd.concat(
        [st.session_state.prediction_history, pd.DataFrame([{"credit_score": credit_score, "predicted_risk_score": average_risk_score}])],
        ignore_index=True
    )
    
    st.session_state.scatter_data = pd.concat(
        [st.session_state.scatter_data, pd.DataFrame([{"credit_score": credit_score, "risk_score": average_risk_score}])],
        ignore_index=True
    )

    st.session_state.latest_features = pd.Series(input_data)

    # Calculate feature contributions (hypothetical approach)
    feature_contributions = {}
    for feature, value in input_data.items():
        contribution = value * (average_risk_score / sum(input_data.values()))
        feature_contributions[feature] = contribution

    # Sort contributions by impact on risk score
    st.session_state.feature_contributions = pd.Series(feature_contributions).sort_values(ascending=False)

col1, col2, col3 = st.columns([2, 2, 2])

# Prediction History
with col1:
    st.subheader("Prediction History")
    st.dataframe(st.session_state.prediction_history.tail(10))

# Model Risk Predictions
with col2:
    st.subheader("Model Risk Predictions")
    for name, score in risk_scores.items():
        st.markdown(f"**{name} Prediction:** `{score:.4f}`")
    st.markdown(f"**Average Risk Score:** `{average_risk_score:.4f}`")

# Model Performance Comparison
with col3:
    st.subheader("Model Performance Comparison")
    model_performance = pd.DataFrame(
        [{"Model": name, "Predicted Risk": f"{score:.4f}"} for name, score in risk_scores.items()]
    )
    st.dataframe(model_performance)

# Risk Score Trends Scatter Plot
st.subheader("Risk Score Trends")
fig = px.scatter(
    st.session_state.scatter_data,
    x="credit_score",
    y="risk_score",
    color_discrete_sequence=["blue"],
    title="Risk Score Trends",
)

# Highlight previous predictions in red and latest in gold
fig.add_trace(
    go.Scatter(
        x=st.session_state.prediction_history["credit_score"],
        y=st.session_state.prediction_history["predicted_risk_score"],
        mode="markers",
        marker=dict(color="red", size=10),
        name="Previous Predictions",
    )
)
fig.add_trace(
    go.Scatter(
        x=[credit_score],
        y=[average_risk_score],
        mode="markers",
        marker=dict(color="gold", size=15, symbol="star"),
        name="Latest Prediction",
    )
)

st.plotly_chart(fig, use_container_width=True)

# Feature Contribution Analysis
st.subheader("Feature Contribution Analysis")
if "feature_contributions" in st.session_state:
    st.write(st.session_state.feature_contributions)

# Risk Score Distribution
st.subheader("Risk Score Distribution")
hist_fig = px.histogram(df, x="risk_score", nbins=20, title="Risk Score Distribution in Dataset")
st.plotly_chart(hist_fig, use_container_width=True)
