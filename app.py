import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Video Game Global Sales Predictor",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Video Game Global Sales Prediction")
st.write(
    "Explore the cleaned video game sales dataset and evaluate "
    "the tuned XGBoost regression model."
)

# ---------------------------------------------------------
# Load project files
# Keep these files in the same folder as app.py
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    return pd.read_csv("video_game_sales_cleaned.csv")

@st.cache_data
def load_test_features():
    return pd.read_csv("X_test_processed.csv")

@st.cache_data
def load_test_target():
    return pd.read_csv("y_test.csv").squeeze("columns")

@st.cache_resource
def load_model():
    return joblib.load("best_xgboost_model.pkl")


required_files = [
    "video_game_sales_cleaned.csv",
    "X_test_processed.csv",
    "y_test.csv",
    "best_xgboost_model.pkl",
]

missing_files = [filename for filename in required_files if not __import__("os").path.exists(filename)]

if missing_files:
    st.error("Some project files are missing. Save them from your Jupyter notebook first:")
    for filename in missing_files:
        st.write(f"- `{filename}`")
    st.info(
        "Add and run the save-files cell shown with this app, then place the generated "
        "files in the same folder as app.py."
    )
    st.stop()

try:
    df = load_dataset()
    X_test = load_test_features()
    y_test = load_test_target()
    model = load_model()

    # Keep the model input columns in exactly the same order as training.
    if hasattr(model, "feature_names_in_"):
        X_test = X_test.reindex(columns=model.feature_names_in_, fill_value=0)

    predictions = np.asarray(model.predict(X_test)).reshape(-1)
    actual = np.asarray(y_test).reshape(-1)

    if len(actual) != len(predictions):
        st.error("The saved test features and target have different row counts.")
        st.stop()

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)

    results = pd.DataFrame({
        "Actual Global Sales": actual,
        "Predicted Global Sales": predictions
    })
    results["Prediction Error"] = (
        results["Actual Global Sales"] - results["Predicted Global Sales"]
    )

except Exception as error:
    st.error(f"Could not load or evaluate the model: {error}")
    st.warning(
        "Check that the model, processed test features, and test target were saved "
        "from the same notebook run."
    )
    st.stop()

# ---------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Dataset", "Model Performance", "Predictions"]
)

# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------
if page == "Dashboard":
    st.header("Project Dashboard")
    st.write(
        "This application presents the video game sales dataset and the performance "
        "of the tuned XGBoost model."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Games in dataset", f"{len(df):,}")
    col2.metric("Features used", f"{X_test.shape[1]:,}")
    col3.metric("Test MAE", f"{mae:.3f}")
    col4.metric("Test R²", f"{r2:.3f}")

    st.subheader("Global Sales Distribution")
    if "Global_Sales" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df["Global_Sales"].dropna(), bins=50)
        ax.set_xlabel("Global Sales (millions)")
        ax.set_ylabel("Number of Games")
        ax.set_title("Distribution of Global Video Game Sales")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("The dataset does not contain a Global_Sales column.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
elif page == "Dataset":
    st.header("Cleaned Dataset")
    st.write(f"Rows: **{len(df):,}** | Columns: **{len(df.columns):,}**")

    search_text = st.text_input("Search column names")
    if search_text:
        visible_columns = [
            col for col in df.columns if search_text.lower() in col.lower()
        ]
        if visible_columns:
            st.dataframe(df[visible_columns], use_container_width=True)
        else:
            st.warning("No column names matched your search.")
    else:
        st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download cleaned dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="video_game_sales_cleaned.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# Model Performance
# ---------------------------------------------------------
elif page == "Model Performance":
    st.header("Tuned XGBoost Model Performance")
    st.caption("Metrics are calculated from the saved test set and model predictions.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
    col3.metric("R² Score", f"{r2:.4f}")

    st.markdown(
        """
        - **MAE:** Average absolute difference between actual and predicted sales.
        - **RMSE:** Gives greater weight to larger prediction errors.
        - **R²:** Indicates how much variation in the test-set sales is explained by the model.
        """
    )

    st.subheader("Actual vs Predicted Global Sales")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predictions, alpha=0.4)
    lower = min(float(np.min(actual)), float(np.min(predictions)))
    upper = max(float(np.max(actual)), float(np.max(predictions)))
    ax.plot([lower, upper], [lower, upper], linestyle="--")
    ax.set_xlabel("Actual Global Sales (millions)")
    ax.set_ylabel("Predicted Global Sales (millions)")
    ax.set_title("Actual vs Predicted Sales")
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------
elif page == "Predictions":
    st.header("Test Set Predictions")
    st.write("Compare the model's predictions with the actual global sales values.")

    st.dataframe(results.head(100), use_container_width=True)

    st.download_button(
        "Download prediction results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="prediction_results.csv",
        mime="text/csv"
    )

    st.subheader("Prediction Error Distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(results["Prediction Error"], bins=40)
    ax.set_xlabel("Actual minus Predicted Sales (millions)")
    ax.set_ylabel("Number of Games")
    ax.set_title("Distribution of Prediction Errors")
    st.pyplot(fig)
    plt.close(fig)

st.sidebar.divider()
st.sidebar.caption("Video Game Sales Prediction Project")
