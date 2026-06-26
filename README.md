# 💳 Credit Card Fraud Detection System using Machine Learning

## 📌 Overview

This project is a Machine Learning-based Credit Card Fraud Detection System developed using Python and Scikit-learn. The system predicts whether a credit card transaction is fraudulent by analyzing transaction features. A Streamlit web application is also developed for easy prediction through CSV upload.

---

## 🎯 Objectives

- Detect fraudulent credit card transactions.
- Compare multiple Machine Learning models.
- Evaluate model performance.
- Deploy the best model using Streamlit.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

## 📂 Project Structure

```
CreditCard-Fraud-Detection/
│
├── dataset/
│   └── creditcard.csv
│
├── models/
│   ├── model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🤖 Models Used

- Logistic Regression
- Decision Tree
- Random Forest (Best Model)

---

## 📊 Model Performance

| Metric | Value |
|---------|--------|
| Accuracy | 99.96% |
| Precision | 94.12% |
| Recall | 81.63% |
| F1 Score | 87.43% |

---

## 🚀 Features

- Data Loading
- Exploratory Data Analysis
- Data Preprocessing
- Model Training
- Model Evaluation
- Streamlit Web Application
- CSV Upload Prediction
- Download Prediction Results

---

## ▶️ How to Run

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---
