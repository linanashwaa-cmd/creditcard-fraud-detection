import streamlit as st
import pandas as pd
import joblib

from src.config import MODEL_PATH, SCALER_PATH

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.sidebar.title("💳 Credit Card Fraud Detection")

st.sidebar.markdown("""
### Project Information

**Algorithm:** Random Forest

**Dataset:** Credit Card Transactions

**Accuracy:** 99.96%

**Developer:** Lina
""")

st.title("💳 Credit Card Fraud Detection System using Random Forest")
st.markdown("""
This application predicts fraudulent credit card transactions using a trained Random Forest Machine Learning model.

Upload a CSV file below to begin prediction.
""")

st.subheader("Dataset Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", "284,807")
col2.metric("Fraud Cases", "492")
col3.metric("Genuine Cases", "284,315")
st.divider()

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df.copy()
    if "Prediction" in X.columns:
        X = X.drop("Prediction", axis=1)
    if "Amount" in X.columns:
        X["Amount"] = scaler.transform(X[["Amount"]])

    predictions = model.predict(X)

    result = df.copy()
    result["Prediction"] = predictions

    fraud_count = (predictions == 1).sum()
    genuine_count = (predictions == 0).sum()

    st.success("Prediction Completed!")

    col1, col2 = st.columns(2)
    col1.error(f"🚨 Fraud Transactions\n\n{fraud_count}")
    col2.success(f"✅ Genuine Transactions\n\n{genuine_count}")


    st.divider()
    st.subheader("Prediction Results")
    st.dataframe(result)

    csv = result.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Prediction Results",
        csv,
        "prediction_results.csv",
        "text/csv"
    )