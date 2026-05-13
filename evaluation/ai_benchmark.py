import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import os

print("Connecting to the database...")
# Connect to the local TimescaleDB instance exposed on port 5432
engine = create_engine('postgresql+psycopg://powerdetect:powerdetect123@localhost:5432/powerdetect_dataset')

print("Fetching data from power_readings...")
# We query the entire power_readings table since it was trimmed to the exact 48h experiment window.
query = """
    SELECT 
        sensor_voltage_mean_v,
        sensor_voltage_peak_v,
        absolute_current_ma,
        variance_current_ma,
        peak_current_ma,
        current_variance_sigma_ma,
        power_mw,
        window_samples,
        attack_state
    FROM power_readings
"""
df = pd.read_sql(query, engine)

print(f"Data loaded: {len(df)} samples.")
print("Class distribution:")
print(df['attack_state'].value_counts())

# Prepare features and target
X = df.drop(columns=['attack_state'])
y = df['attack_state']

# Encode the categorical labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split (80% train, 20% test)
# Stratify ensures the class distribution is maintained in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Testing set:  {len(X_test)} samples")

# Compute sample weights for the training set to handle severe class imbalance
print("\nComputing sample weights to handle extreme imbalance...")
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Initialize XGBoost Classifier
# We use multi:softmax for multi-class classification
print("\nTraining XGBoost model (this may take a moment)...")
clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
clf.fit(X_train, y_train, sample_weight=sample_weights)

# Predict on the test set
print("Evaluating model...")
y_pred = clf.predict(X_test)

# Print classification report
print("\n" + "="*50)
print("             AI BENCHMARK RESULTS")
print("="*50)
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Generate and save the confusion matrix plot
os.makedirs('plots', exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('XGBoost Confusion Matrix - PowerDetect Dataset')
plt.xlabel('Predicted Attack State')
plt.ylabel('Actual Attack State')
plt.tight_layout()
plt.savefig('plots/xgboost_confusion_matrix.png', dpi=300)
print("\nConfusion matrix plot saved to: plots/xgboost_confusion_matrix.png")
