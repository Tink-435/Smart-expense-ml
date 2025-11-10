# preprocess.py
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

def load_and_clean_data(path):
    df = pd.read_csv(path)
    # Example: Check column names – assume they include 'Description', 'Category', 'Amount'
    df = df.dropna(subset=['description', 'category', 'amount'])
    df['description'] = df['description'].astype(str).str.lower()
    df['description'] = df['description'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop])
    )
    # If 'Amount' column exists, ensure numeric:
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna(subset=['amount'])
    return df

def vectorize_data(df):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_text = vectorizer.fit_transform(df['description']).toarray()
    # You may also include 'Amount' as a numeric feature:
    X_amount = df[['amount']].values
    import numpy as np
    X = np.hstack([X_text, X_amount])
    y = df['category']
    return X, y, vectorizer
