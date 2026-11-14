
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Video Game Sales Prediction",
    page_icon="🎮",
    layout="wide"
)

# Load saved files
df = pd.read_csv("video_game_sales_cleaned.csv")
results = pd.read_csv("prediction_results.csv")
metrics = joblib.load("model_metrics.pkl")
model = joblib.load("xgboost_model.pkl")

# Application title
st.title("🎮 Video Game Global Sales Prediction")

st.write(
    "An interactive dashboard for exploring video game sales "
    "and evaluating machine learning predictions."
)

st.divider()

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    ["Dashboard", "Dataset", "Model Performance", "Predictions"]
)

# Dashboard
if page == "Dashboard":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Games", len(df))

    col2.metric("Model", "XGBoost")

    col3.metric("R² Score", f"{metrics['R2']:.4f}")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Global Sales Distribution")

    if "Global_Sales" in df.columns:
        st.bar_chart(
            df["Global_Sales"].value_counts().sort_index().head(20)
        )

# Dataset
elif page == "Dataset":

    st.header("Video Game Sales Dataset")

    st.write("Explore the cleaned dataset and engineered features.")

    st.write("Dataset Shape:", df.shape)

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Cleaned Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="video_game_sales_cleaned.csv",
        mime="text/csv"
    )

# Model Performance
elif page == "Model Performance":

    st.header("XGBoost Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{metrics['MAE']:.4f}")

    col2.metric("RMSE", f"{metrics['RMSE']:.4f}")

    col3.metric("R² Score", f"{metrics['R2']:.4f}")

    st.subheader("Actual vs Predicted Sales")

    st.line_chart(
        results.head(50)[["Actual_Sales", "Predicted_Sales"]]
    )

# Predictions
elif page == "Predictions":

    st.header("Model Predictions")

    st.write("Comparison of actual and predicted global sales.")

    st.dataframe(
        results.head(100),
        use_container_width=True
    )

    st.download_button(
        "Download Prediction Results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="prediction_results.csv",
        mime="text/csv"
    )

st.divider()

st.caption("Video Game Global Sales Prediction | Powered by XGBoost and Streamlit")
