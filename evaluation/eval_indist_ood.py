#!/usr/bin/env python3
"""
Machine Learning Evaluation Pipeline for PowerTrace-IoT
=======================================================
Evaluates in-distribution (12h) and out-of-distribution (48h) performance
using XGBoost with a 20-second rolling mean temporal feature set.

Outputs evaluation metrics to stdout and saves prediction arrays to ./cache/ 
for subsequent figure generation.
"""

import os
import warnings
import pandas as pd
import numpy as np

from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = './data'
CACHE_DIR = './cache'
os.makedirs(CACHE_DIR, exist_ok=True)

RAW_FEATURES = [
    'sensor_voltage_mean_v', 'sensor_voltage_peak_v', 'absolute_current_ma',
    'variance_current_ma', 'peak_current_ma', 'current_variance_sigma_ma',
    'power_mw', 'window_samples'
]
WINDOW = 10  # 20-second context window at 2s sampling rate


def load_dataset(path: str) -> pd.DataFrame:
    """Loads a dataset and parses the event time."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please download it.")
    return pd.read_csv(path, parse_dates=['event_time'])


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes a rolling mean for key electrical metrics to capture temporal context."""
    df = df.copy().sort_values(['node_id', 'event_time']).reset_index(drop=True)
    
    for col in ['absolute_current_ma', 'variance_current_ma', 'power_mw']:
        grp = df.groupby('node_id')[col]
        df[f'{col}_roll_mean'] = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
        
    df['node_id_feat'] = df['node_id'].astype(int)
    return df


def preprocess_data(df: pd.DataFrame, features: list) -> tuple:
    """Prepares feature matrix and labels. Treats data-layer spoofing as benign noise."""
    df = df.copy()
    df['physical_attack'] = df['attack_state'].replace({
        'spoofing_dht': 'none', 
        'spoofing_mpu': 'none'
    })
    df['is_attack'] = (df['physical_attack'] != 'none').astype(int)
    
    X = df[features]
    y_binary = df['is_attack']
    y_physical = df['physical_attack']
    return X, y_binary, y_physical


def main():
    print("="*60)
    print(" PowerTrace-IoT ML Evaluation Pipeline")
    print("="*60)

    # 1. Load Data
    print("\n[1/5] Loading datasets...")
    train_path = os.path.join(BASE_DIR, 'train_12h', 'power_readings_train.csv')
    test_path = os.path.join(BASE_DIR, 'test_48h', 'power_readings_trimmed.csv')
    
    df_train = load_dataset(train_path)
    df_test = load_dataset(test_path)
    print(f"      Train (12h): {len(df_train):,} samples")
    print(f"      Test (48h):  {len(df_test):,} samples")

    # 2. Feature Engineering
    print("\n[2/5] Building temporal features...")
    df_train = build_temporal_features(df_train)
    df_test = build_temporal_features(df_test)
    
    feature_set = RAW_FEATURES + [f'{c}_roll_mean' for c in ['absolute_current_ma', 'variance_current_ma', 'power_mw']] + ['node_id_feat']
    print(f"      Feature set dimensions: {len(feature_set)}")

    X_train, y_bin_train, y_phys_train = preprocess_data(df_train, feature_set)
    X_test, y_bin_test, y_phys_test = preprocess_data(df_test, feature_set)

    # Model Parameters
    xgb_binary_params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'max_depth': 4, 'n_estimators': 150, 'learning_rate': 0.08,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 8,
        'random_state': 42, 'n_jobs': -1
    }

    xgb_attrib_params = {
        'objective': 'multi:softmax', 'eval_metric': 'mlogloss',
        'max_depth': 4, 'n_estimators': 150, 'learning_rate': 0.08,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 6,
        'random_state': 42, 'n_jobs': -1
    }

    # 3. Binary Detection (In-Dist CV)
    print("\n[3/5] Stage 1: Binary Detection (In-Distribution 5-Fold CV)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_aucs, pr_aucs = [], []
    all_y_bin, all_pred_bin = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_bin_train), 1):
        clf = xgb.XGBClassifier(**xgb_binary_params)
        sample_weights = compute_sample_weight('balanced', y_bin_train.iloc[train_idx])
        clf.fit(X_train.iloc[train_idx], y_bin_train.iloc[train_idx], sample_weight=sample_weights)
        
        proba = clf.predict_proba(X_train.iloc[val_idx])[:, 1]
        pred = clf.predict(X_train.iloc[val_idx])
        
        roc_aucs.append(roc_auc_score(y_bin_train.iloc[val_idx], proba))
        pr_aucs.append(average_precision_score(y_bin_train.iloc[val_idx], proba))
        all_y_bin.extend(y_bin_train.iloc[val_idx])
        all_pred_bin.extend(pred)
        
    print(f"      Mean ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
    print(f"      Mean PR-AUC:  {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")

    # 4. Physical Attribution (In-Dist CV)
    print("\n[4/5] Stage 2: Physical Attribution (In-Distribution 5-Fold CV)")
    attack_mask = y_phys_train != 'none'
    X_train_attrib = X_train[attack_mask]
    y_train_attrib_raw = y_phys_train[attack_mask]

    label_encoder = LabelEncoder()
    y_train_attrib = label_encoder.fit_transform(y_train_attrib_raw)
    
    all_y_attrib, all_pred_attrib = [], []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_attrib, y_train_attrib), 1):
        clf_attrib = xgb.XGBClassifier(**{**xgb_attrib_params, 'num_class': len(label_encoder.classes_)})
        sample_weights = compute_sample_weight('balanced', y_train_attrib[train_idx])
        clf_attrib.fit(X_train_attrib.iloc[train_idx], y_train_attrib[train_idx], sample_weight=sample_weights)
        
        all_y_attrib.extend(y_train_attrib[val_idx])
        all_pred_attrib.extend(clf_attrib.predict(X_train_attrib.iloc[val_idx]))

    print("      In-Distribution Attribution Report:")
    print(classification_report(all_y_attrib, all_pred_attrib, target_names=label_encoder.classes_, digits=2, zero_division=0))

    # 5. Out-of-Distribution Evaluation
    print("\n[5/5] Out-of-Distribution Generalization (12h Train -> 48h Eval)")
    
    # Binary
    clf_ood_bin = xgb.XGBClassifier(**xgb_binary_params)
    weights_full_bin = compute_sample_weight('balanced', y_bin_train)
    clf_ood_bin.fit(X_train, y_bin_train, sample_weight=weights_full_bin)
    
    proba_ood = clf_ood_bin.predict_proba(X_test)[:, 1]
    pred_ood = clf_ood_bin.predict(X_test)
    
    print(f"      OOD ROC-AUC: {roc_auc_score(y_bin_test, proba_ood):.4f}")
    print(f"      OOD PR-AUC:  {average_precision_score(y_bin_test, proba_ood):.4f}")

    # Attribution
    clf_ood_attrib = xgb.XGBClassifier(**{**xgb_attrib_params, 'num_class': len(label_encoder.classes_)})
    weights_full_attrib = compute_sample_weight('balanced', y_train_attrib)
    clf_ood_attrib.fit(X_train_attrib, y_train_attrib, sample_weight=weights_full_attrib)

    test_attack_mask = y_phys_test != 'none'
    y_test_attrib = label_encoder.transform(y_phys_test[test_attack_mask])
    pred_test_attrib = clf_ood_attrib.predict(X_test[test_attack_mask])

    print("\n      OOD Attribution Report:")
    print(classification_report(y_test_attrib, pred_test_attrib, target_names=label_encoder.classes_, digits=2, zero_division=0))

    # Save Cache for Figures
    print("\nSaving evaluation cache for figure generation...")
    fpr, tpr, _ = roc_curve(y_bin_test, proba_ood)
    prec, rec, _ = precision_recall_curve(y_bin_test, proba_ood)

    cache_data = {
        'fpr': fpr, 'tpr': tpr, 'prec': prec, 'rec': rec,
        'y_true': np.array(y_bin_test), 'y_proba': proba_ood
    }
    
    for name, array in cache_data.items():
        np.save(os.path.join(CACHE_DIR, f'cache_ood_{name}.npy'), array)

    cm = confusion_matrix(y_test_attrib, pred_test_attrib)
    np.save(os.path.join(CACHE_DIR, 'cache_ood_cm.npy'), cm)
    np.save(os.path.join(CACHE_DIR, 'cache_ood_classes.npy'), label_encoder.classes_)

    fi_df = pd.Series(clf_ood_attrib.feature_importances_, index=feature_set).sort_values(ascending=False)
    fi_df.to_csv(os.path.join(CACHE_DIR, 'cache_feature_importance.csv'), header=['importance'])
    
    print(f"Cache successfully saved to {CACHE_DIR}/")
    print("Run `generate_paper_figures.py` to produce the final plots.")

if __name__ == "__main__":
    main()
