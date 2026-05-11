import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Load the new UPI-based processed dataset
df = pd.read_csv('processed_upi_transactions.csv')

# Select features and label
features = ['location_flag', 'odd_time', 'large_amount', 'business_account', 'amount']
X = df[features]
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(lr_model, 'models/lr_model.pkl')
joblib.dump(rf_model, 'models/rf_model.pkl')

# Save accuracy
with open('models/accuracy.txt', 'w') as f:
    f.write(f'Logistic Regression Accuracy: {lr_acc:.2f}\n')
    f.write(f'Random Forest Accuracy: {rf_acc:.2f}\n')

print("✅ Models trained and saved successfully.")
