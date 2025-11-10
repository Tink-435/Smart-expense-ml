💸 Smart Expense ML  
*Intelligent Expense Categorization & Anomaly Detection using Machine Learning

Overview  
Smart Expense ML is an end-to-end machine learning project that classifies personal transactions (e.g., Food, Travel, Bills) and flags unusual spending patterns.  
It combines TF-IDF vectorization, Logistic Regression, and Isolation Forest, deployed via Streamlit for an interactive experience.

Tech Stack  
Python 3, scikit-learn, pandas, numpy, nltk, joblib, streamlit  
- Model: Logistic Regression  
- Features: TF-IDF (text) + Transaction Amount  
- Anomaly Detection: Isolation Forest  

Workflow  
1. Preprocess Data – Clean & vectorize text using TF-IDF  
2. Train Model – Classify expenses via Logistic Regression  
3. Detect Anomalies – Identify outliers with Isolation Forest  
4. Deploy App – Upload CSV → Predict categories → Download results  

Core Math  
Method : Idea 
TF-IDF : Assigns importance to words in transaction text 
Logistic Regression : Predicts category probabilities 
Isolation Forest : Flags statistical outliers 
