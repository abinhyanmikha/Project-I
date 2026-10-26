
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="My ML Project",
    page_icon="🤖",
    layout="wide"
)

# Website title
st.title("🤖 My Machine Learning Project")
st.write("An interactive website for exploring ML predictions.")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Data Explorer", "Prediction", "About"]
)

# Home page
if page == "Home":
    st.header("Welcome!")
    st.write("Explore my machine learning application.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "ML")
    col2.metric("Status", "Ready")
    col3.metric("Version", "1.0")

# Data Explorer page
elif page == "Data Explorer":
    st.header("Explore Your Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Dataset Statistics")
        st.write(df.describe())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

# Prediction page
elif page == "Prediction":
    st.header("Make a Prediction")

    feature1 = st.number_input("Feature 1", value=5.0)
    feature2 = st.number_input("Feature 2", value=10.0)

    if st.button("Predict"):
        # Example only; replace with your trained model
        result = feature1 + feature2

        st.success(f"Example result: {result}")

# About page
elif page == "About":
    st.header("About This Project")
    st.write(
        "This application demonstrates how Streamlit "
        "can provide a web interface for a machine learning model."
    )
