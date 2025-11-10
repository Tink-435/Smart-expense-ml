# app.py
import streamlit as st
import pandas as pd
import joblib
from anomaly import detect_anomalies

st.title("💸 Smart Expense Categorization & Anomaly Detection")

model = joblib.load('expense_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

uploaded_file = st.file_uploader("Upload your transactions CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview:")
    st.dataframe(df.head())

    if {'description', 'amount'}.issubset(df.columns):
        # Clean description
        df['DescriptionClean'] = df['description'].astype(str).str.lower()
        # Transform
        X_text = vectorizer.transform(df['DescriptionClean']).toarray()
        import numpy as np
        X = np.hstack([X_text, df[['amount']].values])
        preds = model.predict(X)
        df['PredictedCategory'] = le.inverse_transform(preds)
        df = detect_anomalies(df)

        st.write("### Results:")
        st.dataframe(df[['description', 'amount', 'PredictedCategory', 'AnomalyFlag']])

        st.download_button(
            label="⬇️ Download Results as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='expense_results.csv',
            mime='text/csv'
        )
    else:
        st.warning("Ensure your file has both 'Description' and 'Amount' columns.")
