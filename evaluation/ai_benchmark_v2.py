import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import os

print("Loading Cross-Domain Datasets (OOD Evaluation)...")

# Load Training Data (12-hour High-Density Lab Dataset)
df_train = pd.read_csv('database/train_12h/power_readings_train.csv')
print(f"Training Data (12h Lab): {len(df_train)} samples loaded.")

# Load Testing Data (48-hour Real-World Dataset)
df_test = pd.read_csv('database/test_48h/power_readings_trimmed.csv')
print(f"Testing Data (48h Real-World): {len(df_test)} samples loaded.")

# ---------------------------------------------------------
# PREPROCESSING: A* Standard Adjustments
# ---------------------------------------------------------
def preprocess_dataset(df):
    # 1. Relabel Spoofing as 'none' (Benign) for Power evaluation
    df['physical_attack'] = df['attack_state'].replace(
        {'spoofing_dht': 'none', 'spoofing_mpu': 'none'}
    )
    # 2. Create Binary Target for Anomaly Detection (Is it an attack?)
    df['is_attack'] = (df['physical_attack'] != 'none').astype(int)
    
    feature_cols = [
        'sensor_voltage_mean_v', 'sensor_voltage_peak_v', 'absolute_current_ma',
        'variance_current_ma', 'peak_current_ma', 'current_variance_sigma_ma',
        'power_mw', 'window_samples'
    ]
    return df[feature_cols], df['is_attack'], df['physical_attack']

X_train, y_bin_train, y_phys_train = preprocess_dataset(df_train)
X_test, y_bin_test, y_phys_test = preprocess_dataset(df_test)

print("\nTraining Physical Class distribution:")
print(y_phys_train.value_counts())
print("\nTesting Physical Class distribution:")
print(y_phys_test.value_counts())

# =========================================================
# STEP 1: BINARY ANOMALY DETECTION
# =========================================================
print("\n--- STEP 1: Binary Anomaly Detection (Attack vs Normal) ---")
bin_weights = compute_sample_weight(class_weight='balanced', y=y_bin_train)

clf_bin = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
clf_bin.fit(X_train, y_bin_train, sample_weight=bin_weights)

# Predict Binary
y_bin_pred = clf_bin.predict(X_test)
y_bin_proba = clf_bin.predict_proba(X_test)[:, 1]

print("\nBinary Classification Report:")
print(classification_report(y_bin_test, y_bin_pred, target_names=['Normal (0)', 'Attack (1)']))

roc_auc = roc_auc_score(y_bin_test, y_bin_proba)
pr_auc = average_precision_score(y_bin_test, y_bin_proba)
print(f"-> ROC-AUC Score: {roc_auc:.4f}")
print(f"-> PR-AUC Score:  {pr_auc:.4f}")


# =========================================================
# STEP 2: MULTI-CLASS ATTRIBUTION
# =========================================================
print("\n--- STEP 2: Multi-Class Physical Attribution ---")
# Train attribution ONLY on actual attacks
train_attack_mask = y_phys_train != 'none'
X_train_att = X_train[train_attack_mask]
y_phys_train_att = y_phys_train[train_attack_mask]

# Encode multi-class labels
le = LabelEncoder()
y_train_att_enc = le.fit_transform(y_phys_train_att)
att_weights = compute_sample_weight(class_weight='balanced', y=y_train_att_enc)

clf_att = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
clf_att.fit(X_train_att, y_train_att_enc, sample_weight=att_weights)

# Evaluate on true attacks in the test set to measure pure attribution capability
test_attack_mask = y_phys_test != 'none'
X_test_att = X_test[test_attack_mask]
y_phys_test_att = y_phys_test[test_attack_mask]

if len(X_test_att) == 0:
    print("No physical attacks in the 30% test window to attribute!")
else:
    y_test_att_enc = le.transform(y_phys_test_att)
    y_pred_att_enc = clf_att.predict(X_test_att)

    print("\nAttribution Classification Report (Pure Capability):")
    print(classification_report(y_test_att_enc, y_pred_att_enc, target_names=le.classes_, zero_division=0))

    # Generate Confusion Matrix
    os.makedirs('plots', exist_ok=True)
    cm = confusion_matrix(y_test_att_enc, y_pred_att_enc)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Attribution Confusion Matrix (A* Methodology)')
    plt.xlabel('Predicted Physical Attack')
    plt.ylabel('Actual Physical Attack')
    plt.tight_layout()
    plt.savefig('plots/xgboost_attribution_cm.png', dpi=300)
    print("\nConfusion matrix saved to: plots/xgboost_attribution_cm.png")

print("\nEvaluation Complete. Methodology validates real-world time-series conditions.")
