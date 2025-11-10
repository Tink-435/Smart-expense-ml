# anomaly.py
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    # Using amount only for anomaly detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['AnomalyFlag'] = iso.fit_predict(df[['amount']])
    df['AnomalyFlag'] = df['AnomalyFlag'].map({1: 'Normal', -1: 'Anomaly'})
    return df
