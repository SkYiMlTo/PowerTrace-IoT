#!/usr/bin/env python3
"""
Comprehensive figure generation for PowerTrace-IoT paper.
Generates all publication-quality figures for the ICICS 2026 submission.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

ATTACK_COLORS = {
    'none':        '#2196F3',   # blue
    'ddos':        '#F44336',   # red
    'exhaustion':  '#FF9800',   # orange
    'tampering':   '#9C27B0',   # purple
    'spoofing_dht':'#4CAF50',   # green
    'spoofing_mpu':'#009688',   # teal
    'kamikaze':    '#E91E63',   # pink
    'disconnect':  '#607D8B',   # blue-grey
}

ATTACK_LABELS = {
    'none':        'Benign',
    'ddos':        'DDoS',
    'exhaustion':  'Exhaustion',
    'tampering':   'Tampering',
    'spoofing_dht':'Spoofing (DHT)',
    'spoofing_mpu':'Spoofing (MPU)',
    'kamikaze':    'Kamikaze',
    'disconnect':  'Disconnect',
}

BASE = '/Users/bourreauhugo/Documents/2024_bourreau_hugo/07_working_directory/03_Productions/Writings/2.5 dataset/implementation'
OUT  = '/Users/bourreauhugo/Documents/2024_bourreau_hugo/07_working_directory/03_Productions/Writings/2.5 dataset/paper/figures'


import os; os.makedirs(OUT, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading datasets...")
df_train = pd.read_csv(f'{BASE}/database/train_12h/power_readings_train.csv', parse_dates=['event_time'])
df_test  = pd.read_csv(f'{BASE}/database/test_48h/power_readings_trimmed.csv',  parse_dates=['event_time'])

df_all   = pd.concat([df_train, df_test], ignore_index=True)
df_all   = df_all.sort_values('event_time').reset_index(drop=True)

print(f"Train: {len(df_train):,}  |  Test: {len(df_test):,}  |  Total: {len(df_all):,}")

# ─── Figure 1: Power Consumption Timeline (48h test set) ─────────────────────
print("Fig 1: Power timeline (48h)...")
fig, axes = plt.subplots(5, 1, figsize=(7, 6.5), sharex=True)
fig.suptitle('Absolute Current Consumption per Node — 48-Hour Evaluation Dataset', fontsize=10, y=1.01)

nodes = [1, 2, 3, 4, 5]
for ax, nid in zip(axes, nodes):
    df_n = df_test[df_test['node_id'] == nid].copy()
    if df_n.empty:
        continue
    # Plot benign baseline
    benign = df_n[df_n['attack_state'] == 'none']
    ax.plot(benign['event_time'], benign['absolute_current_ma'],
            color='#2196F3', linewidth=0.4, alpha=0.7, zorder=1)
    # Overlay colored attack spans
    for atk, color in ATTACK_COLORS.items():
        if atk == 'none':
            continue
        atk_df = df_n[df_n['attack_state'] == atk]
        if atk_df.empty:
            continue
        ax.scatter(atk_df['event_time'], atk_df['absolute_current_ma'],
                   c=color, s=1.5, zorder=2, alpha=0.85, linewidths=0)
    ax.set_ylabel(f'N{nid}\n(mA)', rotation=0, labelpad=30, va='center')
    ax.set_ylim(0, 180)
    ax.yaxis.set_major_locator(MaxNLocator(3))

axes[-1].set_xlabel('Time (UTC)')
# Legend
patches = [mpatches.Patch(color=v, label=ATTACK_LABELS[k]) for k,v in ATTACK_COLORS.items()]
fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.04),
           frameon=False, markerscale=2)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_power_timeline_48h.pdf')
plt.savefig(f'{OUT}/fig1_power_timeline_48h.png')
plt.close()
print("  -> Saved fig1")

# ─── Figure 2: Box plots — current per attack class ──────────────────────────
print("Fig 2: Box plots per class...")
ORDER = ['none', 'ddos', 'exhaustion', 'tampering', 'spoofing_dht', 'spoofing_mpu', 'kamikaze', 'disconnect']
ORDER_PRESENT = [c for c in ORDER if c in df_all['attack_state'].unique()]
COLORS_LIST   = [ATTACK_COLORS[c] for c in ORDER_PRESENT]
LABELS_LIST   = [ATTACK_LABELS[c] for c in ORDER_PRESENT]

fig, ax = plt.subplots(figsize=(7, 3.5))
data_list = [df_all[df_all['attack_state'] == cls]['absolute_current_ma'].values for cls in ORDER_PRESENT]
bp = ax.boxplot(data_list, patch_artist=True, notch=True, showfliers=True,
                flierprops=dict(marker='.', markersize=1, alpha=0.2),
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8))
for patch, color in zip(bp['boxes'], COLORS_LIST):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_xticks(range(1, len(ORDER_PRESENT)+1))
ax.set_xticklabels(LABELS_LIST, rotation=20, ha='right')
ax.set_ylabel('Absolute Current (mA)')
ax.set_title('Current Consumption Distribution per Attack Class')
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_boxplots_per_class.pdf')
plt.savefig(f'{OUT}/fig2_boxplots_per_class.png')
plt.close()
print("  -> Saved fig2")

# ─── Figure 3: Violin plots — current per class ───────────────────────────────
print("Fig 3: Violin plots...")
fig, ax = plt.subplots(figsize=(7, 3.5))
# Subset — exclude kamikaze/disconnect (too few)
VIOLIN_CLASSES = ['none', 'ddos', 'exhaustion', 'tampering', 'spoofing_dht', 'spoofing_mpu']
df_violin = df_all[df_all['attack_state'].isin(VIOLIN_CLASSES)].copy()
df_violin['class_label'] = df_violin['attack_state'].map(ATTACK_LABELS)

parts = ax.violinplot(
    [df_violin[df_violin['attack_state'] == cls]['absolute_current_ma'].values for cls in VIOLIN_CLASSES],
    positions=range(len(VIOLIN_CLASSES)),
    showmeans=True, showmedians=True, showextrema=False
)
for pc, cls in zip(parts['bodies'], VIOLIN_CLASSES):
    pc.set_facecolor(ATTACK_COLORS[cls])
    pc.set_alpha(0.7)

ax.set_xticks(range(len(VIOLIN_CLASSES)))
ax.set_xticklabels([ATTACK_LABELS[c] for c in VIOLIN_CLASSES])
ax.set_ylabel('Absolute Current (mA)')
ax.set_title('Current Distribution (Violin) per Attack Class')
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_violin_per_class.pdf')
plt.savefig(f'{OUT}/fig3_violin_per_class.png')
plt.close()
print("  -> Saved fig3")

# ─── Figure 4: Mean power per class (bar chart with error bars) ───────────────
print("Fig 4: Mean power bar chart...")
stats_df = df_all.groupby('attack_state')['absolute_current_ma'].agg(['mean','std','count']).reset_index()
stats_df = stats_df[stats_df['attack_state'].isin(ORDER_PRESENT)].set_index('attack_state').loc[ORDER_PRESENT].reset_index()
stats_df['label'] = stats_df['attack_state'].map(ATTACK_LABELS)
# Baseline reference
baseline_mean = stats_df[stats_df['attack_state'] == 'none']['mean'].values[0]

fig, ax = plt.subplots(figsize=(7, 3.2))
bars = ax.bar(range(len(stats_df)), stats_df['mean'],
              yerr=stats_df['std'], capsize=4,
              color=[ATTACK_COLORS[c] for c in stats_df['attack_state']],
              alpha=0.80, edgecolor='white', linewidth=0.5,
              error_kw=dict(elinewidth=0.8, ecolor='#444'))
ax.axhline(baseline_mean, color='#2196F3', linestyle='--', linewidth=0.9, alpha=0.8, label=f'Benign mean ({baseline_mean:.1f} mA)')
ax.set_xticks(range(len(stats_df)))
ax.set_xticklabels(stats_df['label'], rotation=20, ha='right')
ax.set_ylabel('Mean Current (mA)')
ax.set_title('Mean Current Consumption per Attack Class (±1σ)')
ax.legend(loc='upper right')

# Annotate multiplier vs baseline
for i, row in stats_df.iterrows():
    mult = row['mean'] / baseline_mean
    ax.text(i, row['mean'] + row['std'] + 2, f'{mult:.1f}×', ha='center', va='bottom',
            fontsize=7, color='#333')

plt.tight_layout()
plt.savefig(f'{OUT}/fig4_mean_power_per_class.pdf')
plt.savefig(f'{OUT}/fig4_mean_power_per_class.png')
plt.close()
print("  -> Saved fig4")

# ─── Figure 5: Cross-node correlation heatmap ────────────────────────────────
print("Fig 5: Cross-node correlation heatmaps...")
fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
conditions = [
    ('none',       'Benign Baseline'),
    ('ddos',       'DDoS (Coordinated)'),
    ('exhaustion', 'Exhaustion (Coordinated)'),
]
for ax, (cls, title) in zip(axes, conditions):
    subset = df_test[df_test['attack_state'] == cls]
    pivot  = subset.pivot_table(values='absolute_current_ma', index='event_time', columns='node_id')
    if pivot.shape[1] < 2:
        ax.set_visible(False)
        continue
    corr = pivot.corr()
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)
    sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=-0.3, vmax=1.0, mask=mask,
                annot_kws={'size': 7}, linewidths=0.5,
                cbar=False if ax != axes[-1] else True)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Node ID')
    ax.set_ylabel('Node ID' if ax == axes[0] else '')
plt.suptitle('Pairwise Power Correlation Between Nodes', y=1.02, fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_correlation_heatmap.pdf')
plt.savefig(f'{OUT}/fig5_correlation_heatmap.png')
plt.close()
print("  -> Saved fig5")

# ─── Figure 6: Temporal power trace (zoomed attack window) ───────────────────
print("Fig 6: Temporal attack onset windows...")
fig, axes = plt.subplots(2, 2, figsize=(7, 4.5), sharex=False)
axes = axes.flatten()

ZOOM_ATTACKS = ['ddos', 'exhaustion', 'tampering', 'spoofing_dht']
for ax, atk in zip(axes, ZOOM_ATTACKS):
    # Find a contiguous attack window in the test set
    df_atk = df_test[df_test['attack_state'] == atk].copy()
    if df_atk.empty:
        ax.set_visible(False)
        continue
    # Take a 20-minute centered window around first attack
    t_start = df_atk['event_time'].iloc[0] - pd.Timedelta(minutes=5)
    t_end   = df_atk['event_time'].iloc[0]  + pd.Timedelta(minutes=15)
    window  = df_test[(df_test['event_time'] >= t_start) & (df_test['event_time'] <= t_end)]

    for nid in range(1, 6):
        nw = window[window['node_id'] == nid]
        color = plt.cm.tab10(nid - 1)
        label = f'N{nid}'
        benign_w = nw[nw['attack_state'] == 'none']
        atk_w    = nw[nw['attack_state'] == atk]
        ax.plot(benign_w['event_time'], benign_w['absolute_current_ma'],
                color=color, linewidth=0.7, alpha=0.8)
        ax.scatter(atk_w['event_time'], atk_w['absolute_current_ma'],
                   color=color, s=5, zorder=3, label=label)
    ax.set_title(ATTACK_LABELS.get(atk, atk))
    ax.set_ylabel('Current (mA)')
    ax.set_xlabel('')
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(max(0, ymin - 5), min(200, ymax + 10))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.04), frameon=False)
plt.suptitle('Attack Onset Power Dynamics (Zoomed Windows)', y=1.02, fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_attack_onset_windows.pdf')
plt.savefig(f'{OUT}/fig6_attack_onset_windows.png')
plt.close()
print("  -> Saved fig6")

# ─── Figure 7: Dataset composition (pie + timeline) ──────────────────────────
print("Fig 7: Dataset composition...")
counts = df_all['attack_state'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

# Pie chart
ax = axes[0]
labels_pie  = [ATTACK_LABELS[k] for k in counts.index]
colors_pie  = [ATTACK_COLORS[k] for k in counts.index]
wedges, texts, autotexts = ax.pie(
    counts.values, labels=labels_pie, colors=colors_pie,
    autopct='%1.1f%%', startangle=140,
    pctdistance=0.80, textprops={'fontsize': 7},
    wedgeprops={'edgecolor': 'white', 'linewidth': 0.8}
)
for at in autotexts:
    at.set_fontsize(6.5)
ax.set_title('Sample Distribution\n(Train + Test)')

# Stacked bar: dataset sizes per class for train vs test
ax2 = axes[1]
classes = ORDER_PRESENT
train_counts = [len(df_train[df_train['attack_state'] == c]) for c in classes]
test_counts  = [len(df_test [df_test ['attack_state'] == c]) for c in classes]
x = np.arange(len(classes))
ax2.bar(x, train_counts, color=[ATTACK_COLORS[c] for c in classes], alpha=0.9, label='12h Train', width=0.4, align='edge')
ax2.bar(x - 0.4, test_counts, color=[ATTACK_COLORS[c] for c in classes], alpha=0.45, label='48h Test', width=0.4, align='edge', hatch='//')
ax2.set_xticks(x - 0.2)
ax2.set_xticklabels([ATTACK_LABELS[c] for c in classes], rotation=25, ha='right')
ax2.set_ylabel('Sample Count')
ax2.set_title('Sample Count per Class\n(Train vs. Test Split)')
ax2.legend(frameon=False)
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_dataset_composition.pdf')
plt.savefig(f'{OUT}/fig7_dataset_composition.png')
plt.close()
print("  -> Saved fig7")

# ─── Figure 8: OOD Confusion Matrix (XGBoost Attribution) ────────────────────
print("Fig 8: Confusion matrix...")
# Reproduce classification from saved CSV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

RAW_FEATS = ['sensor_voltage_mean_v','sensor_voltage_peak_v','absolute_current_ma',
             'variance_current_ma','peak_current_ma','current_variance_sigma_ma','power_mw','window_samples']
WINDOW = 10

def add_rolling_features(df):
    df = df.copy().sort_values(['node_id', 'event_time']).reset_index(drop=True)
    for col in ['absolute_current_ma', 'variance_current_ma', 'power_mw']:
        grp = df.groupby('node_id')[col]
        df[f'{col}_roll_mean'] = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
        df[f'{col}_roll_std']  = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(0))
        df[f'{col}_roll_max']  = grp.transform(lambda x: x.rolling(WINDOW, min_periods=1).max())
        df[f'{col}_diff']      = grp.transform(lambda x: x.diff().fillna(0))
    df['node_id_feat'] = df['node_id'].astype(int)
    return df

ROLL_FEATS = (RAW_FEATS +
              [f'{c}_{s}' for c in ['absolute_current_ma', 'variance_current_ma', 'power_mw']
               for s in ['roll_mean', 'roll_std', 'roll_max', 'diff']] +
              ['node_id_feat'])

df_tr_roll = add_rolling_features(df_train)
df_te_roll = add_rolling_features(df_test)

def preprocess(df):
    df = df.copy()
    df['physical_attack'] = df['attack_state'].replace({'spoofing_dht': 'none', 'spoofing_mpu': 'none'})
    df['is_attack'] = (df['physical_attack'] != 'none').astype(int)
    return df[ROLL_FEATS], df['is_attack'], df['physical_attack']

X_tr, y_bin_tr, y_ph_tr = preprocess(df_tr_roll)
X_te, y_bin_te, y_ph_te = preprocess(df_te_roll)

train_mask = y_ph_tr != 'none'
X_tr_att, y_tr_att = X_tr[train_mask], y_ph_tr[train_mask]
le = LabelEncoder()
y_tr_enc = le.fit_transform(y_tr_att)
clf = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_),
                         max_depth=6, n_estimators=300, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                         random_state=42, n_jobs=-1, eval_metric='mlogloss')
clf.fit(X_tr_att, y_tr_enc, sample_weight=compute_sample_weight('balanced', y_tr_enc))

test_mask = y_ph_te != 'none'
X_te_att  = X_te[test_mask]
y_te_att  = y_ph_te[test_mask]
y_te_enc  = le.transform(y_te_att)
y_pred    = clf.predict(X_te_att)

cm = confusion_matrix(y_te_enc, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)


fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, linecolor='white', ax=ax,
            vmin=0, vmax=1,
            annot_kws={'fontsize': 9})
# Overlay raw counts
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        ax.text(j+0.5, i+0.72, f'n={cm[i,j]}',
                ha='center', va='center', fontsize=6, color='#444')
ax.set_xlabel('Predicted Attack Class')
ax.set_ylabel('Ground-Truth Attack Class')
ax.set_title('Physical Attack Attribution\nConfusion Matrix (Cross-Domain Evaluation)', pad=12)
plt.tight_layout()
plt.savefig(f'{OUT}/fig8_confusion_matrix.pdf')
plt.savefig(f'{OUT}/fig8_confusion_matrix.png')
plt.close()
print("  -> Saved fig8")

# ─── Figure 9: ROC curve (Binary detection) ───────────────────────────────────
print("Fig 9: ROC curve...")
from sklearn.metrics import roc_curve, auc, precision_recall_curve

clf_bin = xgb.XGBClassifier(objective='binary:logistic', max_depth=6,
                              n_estimators=300, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                              random_state=42, n_jobs=-1, eval_metric='logloss')
clf_bin.fit(X_tr, y_bin_tr, sample_weight=compute_sample_weight('balanced', y_bin_tr))
y_score = clf_bin.predict_proba(X_te)[:, 1]

fpr, tpr, _ = roc_curve(y_bin_te, y_score)
roc_auc     = auc(fpr, tpr)
prec, rec, _= precision_recall_curve(y_bin_te, y_score)
pr_auc      = auc(rec, prec)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
# ROC
ax = axes[0]
ax.plot(fpr, tpr, color='#F44336', linewidth=1.5, label=f'XGBoost (AUC = {roc_auc:.4f})')
ax.plot([0,1],[0,1], 'k--', linewidth=0.7, label='Random')
ax.fill_between(fpr, tpr, alpha=0.08, color='#F44336')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Binary Anomaly Detection')
ax.legend(loc='lower right', frameon=False)

# PR
ax = axes[1]
ax.plot(rec, prec, color='#FF9800', linewidth=1.5, label=f'XGBoost (AP = {pr_auc:.4f})')
baseline_pr = y_bin_te.mean()
ax.axhline(baseline_pr, color='k', linestyle='--', linewidth=0.7, label=f'Chance ({baseline_pr:.4f})')
ax.fill_between(rec, prec, alpha=0.08, color='#FF9800')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision–Recall Curve — Binary Anomaly Detection')
ax.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.savefig(f'{OUT}/fig9_roc_pr_curves.pdf')
plt.savefig(f'{OUT}/fig9_roc_pr_curves.png')
plt.close()
print("  -> Saved fig9")

# ─── Figure 10: Training dataset power timeline (12h) ─────────────────────────
print("Fig 10: Training timeline (12h)...")
fig, axes = plt.subplots(5, 1, figsize=(7, 5.5), sharex=True)
fig.suptitle('Absolute Current Consumption per Node — 12-Hour Training Dataset', fontsize=10, y=1.01)
for ax, nid in zip(axes, nodes):
    df_n = df_train[df_train['node_id'] == nid].copy()
    if df_n.empty:
        continue
    benign = df_n[df_n['attack_state'] == 'none']
    ax.plot(benign['event_time'], benign['absolute_current_ma'],
            color='#2196F3', linewidth=0.4, alpha=0.7, zorder=1)
    for atk, color in ATTACK_COLORS.items():
        if atk == 'none':
            continue
        atk_df = df_n[df_n['attack_state'] == atk]
        if atk_df.empty:
            continue
        ax.scatter(atk_df['event_time'], atk_df['absolute_current_ma'],
                   c=color, s=1.5, zorder=2, alpha=0.85, linewidths=0)
    ax.set_ylabel(f'N{nid}\n(mA)', rotation=0, labelpad=30, va='center')
    ax.set_ylim(0, 180)
    ax.yaxis.set_major_locator(MaxNLocator(3))
axes[-1].set_xlabel('Time (UTC)')
patches = [mpatches.Patch(color=v, label=ATTACK_LABELS[k]) for k,v in ATTACK_COLORS.items()]
fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.04), frameon=False)
plt.tight_layout()
plt.savefig(f'{OUT}/fig10_power_timeline_12h.pdf')
plt.savefig(f'{OUT}/fig10_power_timeline_12h.png')
plt.close()
print("  -> Saved fig10")

print("\nAll figures saved to:", OUT)
print("Files:")
for f in sorted(os.listdir(OUT)):
    size = os.path.getsize(f'{OUT}/{f}')
    print(f"  {f:45s}  {size/1024:.1f} KB")
