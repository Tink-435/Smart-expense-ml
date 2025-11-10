print("🚀 Starting model training script...")

from preprocess import load_and_clean_data, vectorize_data
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

print("✅ Imports successful!")

# Load and preprocess data
df = load_and_clean_data('transactions.csv')
print("✅ Data loaded:", df.shape)

# Vectorize text and amount features
X, y, vectorizer = vectorize_data(df)
print("✅ Data vectorized:", X.shape, len(y))

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200, multi_class='auto', solver='lbfgs')
model.fit(X_train, y_train)
print("✅ Model trained!")

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save trained components
joblib.dump(model, 'expense_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("✅ Model and vectorizer saved successfully!")
