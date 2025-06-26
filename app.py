import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load model
model = pickle.load(open("fraud_model.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection App")

# File uploader
file = st.file_uploader("Upload transaction CSV file", type=["csv"])

if file:
    data = pd.read_csv(file)

    st.write("✅ Uploaded Data:")
    st.dataframe(data.head())

    # 🛠️ Drop 'Class' column if it exists
    if 'Class' in data.columns:
        data.drop('Class', axis=1, inplace=True)

    # Preprocess
    scaler = StandardScaler()
    data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Predict
    predictions = model.predict(data)
    data['Prediction'] = predictions

    st.write("🎯 Prediction Results:")
    st.dataframe(data)

    frauds = data[data['Prediction'] == 1]
    if len(frauds) > 0:
        st.error(f"⚠️ {len(frauds)} Fraudulent Transactions Detected!")
    else:
        st.success("✅ No frauds found. All good!")
