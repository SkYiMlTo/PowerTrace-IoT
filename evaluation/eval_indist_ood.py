#!/usr/bin/env python3
"""
Improved in-distribution 12h evaluation with temporal rolling features.
DDoS vs Exhaustion are nearly identical in mean current but differ in
temporal variance patterns — rolling features expose this.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

BASE = './data'

print("Loading datasets...")
df_train = pd.read_csv(f'{BASE}/train_12h/power_readings_train.csv',
                       parse_dates=['event_time'])
df_test  = pd.read_csv(f'{BASE}/test_48h/power_readings_trimmed.csv',
                       parse_dates=['event_time'])
print(f"  12h (train): {len(df_train):,}   48h (test): {len(df_test):,}")

RAW_FEATS = ['sensor_voltage_mean_v', 'sensor_voltage_peak_v', 'absolute_current_ma',
             'variance_current_ma', 'peak_current_ma', 'current_variance_sigma_ma',
             'power_mw', 'window_samples']

WINDOW = 10   # readings = 20 seconds of context per node

def add_rolling_features(df):
    """
    Add per-node rolling statistics over the last WINDOW readings.
    Sorted by event_time within each node before rolling.
    """
    df = df.copy().sort_values(['node_id', 'event_time']).reset_index(drop=True)

    for col in ['absolute_current_ma', 'variance_current_ma', 'power_mw']:
        grp = df.groupby('node_id')[col]
        df[f'{col}_roll_mean'] = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
        df[f'{col}_roll_std']  = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(0))
        df[f'{col}_roll_max']  = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).max())
        df[f'{col}_diff']      = grp.transform(lambda x: x.diff().fillna(0))

    # node_id as a numeric feature (captures per-node baseline offsets)
    df['node_id_feat'] = df['node_id'].astype(int)

    return df

ROLL_FEATS = (RAW_FEATS +
              [f'{c}_{s}'
               for c in ['absolute_current_ma', 'variance_current_ma', 'power_mw']
               for s in ['roll_mean', 'roll_std', 'roll_max', 'diff']] +
              ['node_id_feat'])

print(f"Feature set: {len(ROLL_FEATS)} features")

print("Building rolling features (train)...")
df_tr = add_rolling_features(df_train)
print("Building rolling features (test)...")
df_te = add_rolling_features(df_test)

def preprocess(df):
    df = df.copy()
    df['physical_attack'] = df['attack_state'].replace(
        {'spoofing_dht': 'none', 'spoofing_mpu': 'none'})
    df['is_attack'] = (df['physical_attack'] != 'none').astype(int)
    return df[ROLL_FEATS], df['is_attack'], df['physical_attack']

X_tr, y_bin_tr, y_ph_tr = preprocess(df_tr)
X_te, y_bin_te, y_ph_te = preprocess(df_te)

# ─── Stage 1: Binary 5-Fold CV ────────────────────────────────────────────────
print("\n" + "="*60)
print("  STAGE 1 BINARY — 5-Fold CV on 12h")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_aucs, pr_aucs, all_y, all_proba, all_pred = [], [], [], [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_bin_tr), 1):
    Xf_tr, Xf_va = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
    yf_tr, yf_va = y_bin_tr.iloc[tr_idx], y_bin_tr.iloc[va_idx]
    clf = xgb.XGBClassifier(objective='binary:logistic', max_depth=6,
                             n_estimators=300, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5,
                             random_state=42, n_jobs=-1, eval_metric='logloss')
    w = compute_sample_weight('balanced', yf_tr)
    clf.fit(Xf_tr, yf_tr, sample_weight=w)
    proba = clf.predict_proba(Xf_va)[:, 1]
    pred  = clf.predict(Xf_va)
    roc_aucs.append(roc_auc_score(yf_va, proba))
    pr_aucs.append(average_precision_score(yf_va, proba))
    all_y.extend(yf_va); all_proba.extend(proba); all_pred.extend(pred)
    print(f"  Fold {fold}: ROC-AUC={roc_aucs[-1]:.4f}  PR-AUC={pr_aucs[-1]:.4f}")

print(f"\n  ROC-AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
print(f"  PR-AUC:  {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}")
print(classification_report(all_y, all_pred, target_names=['Normal', 'Attack']))

# ─── Stage 2: Attribution 5-Fold CV ───────────────────────────────────────────
print("\n" + "="*60)
print("  STAGE 2 ATTRIBUTION — 5-Fold CV on 12h (attack samples only)")
print("="*60)

att_mask  = y_ph_tr != 'none'
X_att     = X_tr[att_mask]
y_att_raw = y_ph_tr[att_mask]
le = LabelEncoder()
y_att_enc = le.fit_transform(y_att_raw)
print(f"  Classes: {list(le.classes_)}")
print(f"  Attack samples: {len(y_att_enc)}")

skf_att = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_y_att, all_pred_att = [], []

for fold, (tr_idx, va_idx) in enumerate(skf_att.split(X_att, y_att_enc), 1):
    Xa_tr, Xa_va = X_att.iloc[tr_idx], X_att.iloc[va_idx]
    ya_tr, ya_va = y_att_enc[tr_idx], y_att_enc[va_idx]
    clf_att = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_),
                                  max_depth=6, n_estimators=300, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  min_child_weight=3,
                                  random_state=42, n_jobs=-1, eval_metric='mlogloss')
    w = compute_sample_weight('balanced', ya_tr)
    clf_att.fit(Xa_tr, ya_tr, sample_weight=w)
    all_y_att.extend(ya_va)
    all_pred_att.extend(clf_att.predict(Xa_va))
    print(f"  Fold {fold} done.")

print("\n  Attribution Report (12h CV):")
print(classification_report(all_y_att, all_pred_att, target_names=le.classes_))

# ─── OOD: Full 12h → 48h ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("  OOD GENERALISATION: Full 12h Model → 48h Trace")
print("="*60)

clf_ood = xgb.XGBClassifier(objective='binary:logistic', max_depth=6,
                              n_estimators=300, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              min_child_weight=5,
                              random_state=42, n_jobs=-1, eval_metric='logloss')
w_full = compute_sample_weight('balanced', y_bin_tr)
clf_ood.fit(X_tr, y_bin_tr, sample_weight=w_full)
proba_ood = clf_ood.predict_proba(X_te)[:, 1]
pred_ood  = clf_ood.predict(X_te)
print(f"  ROC-AUC (OOD): {roc_auc_score(y_bin_te, proba_ood):.4f}")
print(f"  PR-AUC  (OOD): {average_precision_score(y_bin_te, proba_ood):.4f}")
print(classification_report(y_bin_te, pred_ood, target_names=['Normal', 'Attack']))

clf_att_ood = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_),
                                  max_depth=6, n_estimators=300, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, n_jobs=-1, eval_metric='mlogloss')
clf_att_ood.fit(X_att, y_att_enc, sample_weight=compute_sample_weight('balanced', y_att_enc))

te_att_mask  = y_ph_te != 'none'
X_te_att     = X_te[te_att_mask]
y_te_att_enc = le.transform(y_ph_te[te_att_mask])
y_pred_att   = clf_att_ood.predict(X_te_att)
print("  Attribution Report (OOD):")
print(classification_report(y_te_att_enc, y_pred_att, target_names=le.classes_, zero_division=0))
print("Done.")
