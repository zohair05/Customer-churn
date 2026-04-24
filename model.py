import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import clean_data

# Create necessary directories
os.makedirs('dataset', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('model', exist_ok=True)

# 1. Load and Clean Data
dataset_path = 'dataset/Telco-Customer-Churn.csv'
print(f"Loading dataset from {dataset_path}...")

try:
    df = pd.read_csv(dataset_path)
except FileNotFoundError:
    print(f"❌ Error: Could not find the dataset.")
    print(f"Please ensure your Kaggle CSV is placed inside the 'dataset' folder and named 'Telco-Customer-Churn.csv'.")
    sys.exit(1)

# Clean the data
df = clean_data(df)

# 2. Exploratory Data Analysis (EDA)
print("Generating EDA visualizations...")
# Correlation Heatmap (Numeric features only)
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png")
plt.close()

# 3. Preprocessing for Modeling
print("Encoding features...")
X = df.drop('Churn', axis=1)
y = df['Churn'].map({'Yes': 1, 'No': 0}) # Encode target

# One-hot encoding for categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Save the expected column names for the Streamlit app to ensure alignment
joblib.dump(X_encoded.columns.tolist(), 'model/model_columns.pkl')

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 4. Model Building & Hyperparameter Tuning
print("Training Logistic Regression (Baseline)...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

print("Training & Tuning Random Forest...")
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

# 5. Evaluation
print("\n--- Logistic Regression Results ---")
print(classification_report(y_test, lr_model.predict(X_test)))

print("\n--- Random Forest (Tuned) Results ---")
y_pred_rf = best_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Generate Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Churn")
plt.ylabel("Actual Churn")
plt.savefig("output/confusion_matrix.png")
plt.close()

# 6. Save Best Model
joblib.dump(best_rf, 'model/churn_model.pkl')
print("\n✅ Training complete! Model and columns saved to 'model/' directory.")