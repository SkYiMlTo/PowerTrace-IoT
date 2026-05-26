#!/usr/bin/env python3
"""
PowerTrace-IoT Figure Generation
=================================
Generates all publication-quality figures for the paper:
  "PowerTrace-IoT: An Empirical Power-Consumption Dataset for
   Physical-Layer Attack Detection in Distributed IoT Sensor Networks"

Data characterisation figures (Fig 1–7, 10) are generated from raw CSV
traces.  Evaluation figures (Fig 8, 9, 11) are generated from the NumPy
caches written by ``eval_indist_ood.py``; run that script first.

Usage (from the ``evaluation/`` directory):
    python generate_paper_figures.py

Outputs are written to ``./output_figures/`` as PDF files.
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import auc as sk_auc

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = './data'
CACHE_DIR = './cache'
OUT_DIR   = './output_figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          9,
    'axes.labelsize':     9,
    'axes.titlesize':     10,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

ATTACK_COLORS = {
    'none':         '#2196F3',
    'ddos':         '#F44336',
    'exhaustion':   '#FF9800',
    'tampering':    '#9C27B0',
    'spoofing_dht': '#4CAF50',
    'spoofing_mpu': '#009688',
    'kamikaze':     '#E91E63',
    'disconnect':   '#607D8B',
}

ATTACK_LABELS = {
    'none':         'Benign',
    'ddos':         'DDoS',
    'exhaustion':   'Exhaustion',
    'tampering':    'Tampering',
    'spoofing_dht': 'Spoofing (DHT)',
    'spoofing_mpu': 'Spoofing (MPU)',
    'kamikaze':     'Kamikaze',
    'disconnect':   'Disconnect',
}

ORDER = [
    'none', 'ddos', 'exhaustion', 'tampering',
    'spoofing_dht', 'spoofing_mpu', 'kamikaze', 'disconnect',
]

CLASS_LABELS = {
    'ddos': 'DDoS', 'exhaustion': 'Exhaustion', 'tampering': 'Tampering',
    'kamikaze': 'Kamikaze', 'disconnect': 'Disconnect',
}

FEAT_LABELS = {
    'sensor_voltage_mean_v':           'Voltage Mean (V)',
    'sensor_voltage_peak_v':           'Voltage Peak (V)',
    'absolute_current_ma':             'Abs. Current (mA)',
    'variance_current_ma':             'Variance Current (mA)',
    'peak_current_ma':                 'Peak Current (mA)',
    'current_variance_sigma_ma':       'Current σ (mA)',
    'power_mw':                        'Power (mW)',
    'window_samples':                  'Window Samples',
    'absolute_current_ma_roll_mean':   'Roll. Mean — Abs. Current',
    'variance_current_ma_roll_mean':   'Roll. Mean — Variance',
    'power_mw_roll_mean':              'Roll. Mean — Power',
    'node_id_feat':                    'Node ID',
}


def _save(name: str) -> None:
    plt.savefig(os.path.join(OUT_DIR, f'{name}.pdf'))
    plt.close()
    print(f'  -> {name}.pdf')


# ─── Data loading ──────────────────────────────────────────────────────────────
def load_datasets() -> tuple:
    """Loads the training and evaluation CSVs from ``./data/``."""
    print('Loading raw datasets...')
    train_path = os.path.join(BASE_DIR, 'train_12h', 'power_readings_train.csv')
    test_path  = os.path.join(BASE_DIR, 'test_48h', 'power_readings_trimmed.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f'Datasets not found under {BASE_DIR}/. '
            'Download from Zenodo (https://doi.org/10.5281/zenodo.20347924) '
            'and place them as described in the README.'
        )

    df_train = pd.read_csv(train_path, parse_dates=['event_time'])
    df_test  = pd.read_csv(test_path,  parse_dates=['event_time'])
    df_all   = (pd.concat([df_train, df_test], ignore_index=True)
                  .sort_values('event_time')
                  .reset_index(drop=True))

    print(f'  Train: {len(df_train):,}  |  Test: {len(df_test):,}  |  Total: {len(df_all):,}')
    return df_train, df_test, df_all


# ─── Data characterisation figures ────────────────────────────────────────────
def generate_data_figures(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    df_all:   pd.DataFrame,
) -> None:
    """Generates Figures 1–7 and 10 from raw power traces."""

    nodes         = sorted(df_all['node_id'].unique())
    order_present = [c for c in ORDER if c in df_all['attack_state'].unique()]
    colors_list   = [ATTACK_COLORS[c] for c in order_present]
    labels_list   = [ATTACK_LABELS[c] for c in order_present]

    # Shared legend patches (reused across timeline figures).
    legend_patches = [
        mpatches.Patch(color=ATTACK_COLORS[k], label=ATTACK_LABELS[k])
        for k in ORDER
    ]

    # ── Fig 1: 48-hour power timeline ──────────────────────────────────────
    print('Generating Fig 1: 48h Power Timeline...')
    fig, axes = plt.subplots(5, 1, figsize=(7, 4.8), sharex=True)
    fig.suptitle(
        'Absolute Current Consumption per Node — 48-Hour Evaluation Dataset',
        fontsize=10, y=1.01,
    )

    for ax, nid in zip(axes, nodes):
        df_n   = df_test[df_test['node_id'] == nid].copy()
        if df_n.empty:
            continue
        benign = df_n[df_n['attack_state'] == 'none']
        ax.plot(benign['event_time'], benign['absolute_current_ma'],
                color='#2196F3', linewidth=0.4, alpha=0.7, zorder=1)
        for atk, color in ATTACK_COLORS.items():
            if atk == 'none':
                continue
            atk_df = df_n[df_n['attack_state'] == atk]
            if not atk_df.empty:
                ax.scatter(atk_df['event_time'], atk_df['absolute_current_ma'],
                           c=color, s=1.5, zorder=2, alpha=0.85, linewidths=0)
        ax.set_ylabel(f'N{nid}\n(mA)', rotation=0, labelpad=30, va='center')
        ax.set_ylim(0, 180)
        ax.yaxis.set_major_locator(MaxNLocator(3))

    axes[-1].set_xlabel('Time (UTC)')
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.0), frameon=False, markerscale=2)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save('fig1_power_timeline_48h')

    # ── Fig 2: Box plots per attack class ──────────────────────────────────
    print('Generating Fig 2: Class Box Plots...')
    fig, ax = plt.subplots(figsize=(7, 3.5))
    data_list = [
        df_all[df_all['attack_state'] == cls]['absolute_current_ma'].values
        for cls in order_present
    ]
    bp = ax.boxplot(
        data_list, patch_artist=True, notch=True, showfliers=True,
        flierprops=dict(marker='.', markersize=1, alpha=0.2),
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(order_present) + 1))
    ax.set_xticklabels(labels_list, rotation=20, ha='right')
    ax.set_ylabel('Absolute Current (mA)')
    ax.set_title('Current Consumption Distribution per Attack Class')
    plt.tight_layout()
    _save('fig2_boxplots_per_class')

    # ── Fig 3: Violin plots per attack class ───────────────────────────────
    print('Generating Fig 3: Class Violin Plots...')
    fig, ax = plt.subplots(figsize=(7, 3.8))
    parts = ax.violinplot(
        [df_all[df_all['attack_state'] == cls]['absolute_current_ma'].values
         for cls in ORDER],
        positions=range(len(ORDER)),
        showmeans=True, showmedians=True, showextrema=False,
    )
    for pc, cls in zip(parts['bodies'], ORDER):
        pc.set_facecolor(ATTACK_COLORS[cls])
        pc.set_alpha(0.7)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([ATTACK_LABELS[c] for c in ORDER], rotation=25, ha='right')
    ax.set_ylabel('Absolute Current (mA)')
    ax.set_title('Current Distribution (Violin) per Attack Class')
    plt.tight_layout()
    _save('fig3_violin_per_class')

    # ── Fig 4: Mean power bar chart with ±1σ error bars ────────────────────
    print('Generating Fig 4: Mean Power Bar Chart...')
    stats_df = (
        df_all.groupby('attack_state')['absolute_current_ma']
        .agg(['mean', 'std'])
        .reset_index()
    )
    stats_df = (
        stats_df[stats_df['attack_state'].isin(order_present)]
        .set_index('attack_state').loc[order_present].reset_index()
    )
    baseline_mean = stats_df[stats_df['attack_state'] == 'none']['mean'].values[0]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(
        range(len(stats_df)), stats_df['mean'],
        yerr=stats_df['std'], capsize=4,
        color=[ATTACK_COLORS[c] for c in stats_df['attack_state']],
        alpha=0.80, edgecolor='white', linewidth=0.5,
        error_kw=dict(elinewidth=0.8, ecolor='#444'),
    )
    ax.axhline(baseline_mean, color='#2196F3', linestyle='--', linewidth=0.9,
               alpha=0.8, label=f'Benign mean ({baseline_mean:.1f} mA)')
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels([ATTACK_LABELS[c] for c in stats_df['attack_state']],
                       rotation=20, ha='right')
    ax.set_ylabel('Mean Current (mA)')
    ax.set_title('Mean Current Consumption per Attack Class (±1σ)')
    ax.legend(loc='upper right')
    for i, row in stats_df.iterrows():
        mult = row['mean'] / baseline_mean
        ax.text(i, row['mean'] + row['std'] + 2, f'{mult:.1f}×',
                ha='center', va='bottom', fontsize=7, color='#333')
    plt.tight_layout()
    _save('fig4_mean_power_per_class')

    # ── Fig 5: Cross-node current-correlation heatmaps ─────────────────────
    print('Generating Fig 5: Correlation Heatmaps...')
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))
    conditions = [
        ('none',       'Benign Baseline'),
        ('ddos',       'DDoS (Coordinated)'),
        ('exhaustion', 'Exhaustion (Coordinated)'),
    ]
    for ax, (cls, title) in zip(axes, conditions):
        subset = df_test[df_test['attack_state'] == cls]
        pivot  = subset.pivot_table(
            values='absolute_current_ma', index='event_time', columns='node_id'
        )
        if pivot.shape[1] < 2:
            ax.set_visible(False)
            continue
        corr = pivot.corr()
        mask = np.zeros_like(corr, dtype=bool)
        np.fill_diagonal(mask, True)
        sns.heatmap(corr, ax=ax, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=-0.3, vmax=1.0, mask=mask,
                    annot_kws={'size': 7}, linewidths=0.5,
                    cbar=(ax == axes[-1]))
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Node ID')
        ax.set_ylabel('Node ID' if ax == axes[0] else '')
    plt.suptitle('Pairwise Power Correlation Between Nodes', y=1.02, fontsize=10)
    plt.tight_layout()
    _save('fig5_correlation_heatmap')

    # ── Fig 6: Attack-onset zoomed windows ─────────────────────────────────
    print('Generating Fig 6: Attack Onset Trace...')
    fig, axes = plt.subplots(2, 2, figsize=(7, 4.5), sharex=False)
    axes = axes.flatten()
    zoom_attacks = ['ddos', 'exhaustion', 'tampering', 'spoofing_dht']

    for ax, atk in zip(axes, zoom_attacks):
        df_atk = df_test[df_test['attack_state'] == atk].copy()
        if df_atk.empty:
            ax.set_visible(False)
            continue
        t_start = df_atk['event_time'].iloc[0] - pd.Timedelta(minutes=5)
        t_end   = df_atk['event_time'].iloc[0] + pd.Timedelta(minutes=15)
        window  = df_test[
            (df_test['event_time'] >= t_start) & (df_test['event_time'] <= t_end)
        ]
        for nid in sorted(df_test['node_id'].unique()):
            nw       = window[window['node_id'] == nid]
            color    = plt.cm.tab10(nid - 1)
            benign_w = nw[nw['attack_state'] == 'none']
            atk_w    = nw[nw['attack_state'] == atk]
            ax.plot(benign_w['event_time'], benign_w['absolute_current_ma'],
                    color=color, linewidth=0.7, alpha=0.8)
            ax.scatter(atk_w['event_time'], atk_w['absolute_current_ma'],
                       color=color, s=5, zorder=3, label=f'N{nid}')
        ax.set_title(ATTACK_LABELS.get(atk, atk))
        ax.set_ylabel('Current (mA)')
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(max(0, ymin - 5), min(200, ymax + 10))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save('fig6_attack_onset_windows')

    # ── Fig 7: Dataset composition ─────────────────────────────────────────
    print('Generating Fig 7: Dataset Composition...')
    counts = df_all['attack_state'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=[ATTACK_LABELS[k] for k in counts.index],
        colors=[ATTACK_COLORS[k] for k in counts.index],
        autopct='%1.1f%%', startangle=140,
        pctdistance=0.80, textprops={'fontsize': 7},
        wedgeprops={'edgecolor': 'white', 'linewidth': 0.8},
    )
    for at in autotexts:
        at.set_fontsize(6.5)
    ax.set_title('Sample Distribution\n(Train + Test)')

    ax2 = axes[1]
    train_counts = [len(df_train[df_train['attack_state'] == c]) for c in order_present]
    test_counts  = [len(df_test[df_test['attack_state'] == c]) for c in order_present]
    x = np.arange(len(order_present))
    ax2.bar(x, train_counts,
            color=[ATTACK_COLORS[c] for c in order_present],
            alpha=0.9, label='12h Train', width=0.4, align='edge')
    ax2.bar(x - 0.4, test_counts,
            color=[ATTACK_COLORS[c] for c in order_present],
            alpha=0.45, label='48h Test', width=0.4, align='edge', hatch='//')
    ax2.set_xticks(x - 0.2)
    ax2.set_xticklabels([ATTACK_LABELS[c] for c in order_present],
                        rotation=25, ha='right')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Sample Count per Class')
    ax2.legend(frameon=False)
    plt.tight_layout()
    _save('fig7_dataset_composition')

    # ── Fig 10: 12-hour training timeline ──────────────────────────────────
    print('Generating Fig 10: 12h Power Timeline...')
    fig, axes = plt.subplots(5, 1, figsize=(7, 5.5), sharex=True)
    fig.suptitle(
        'Absolute Current Consumption per Node — 12-Hour Training Dataset',
        fontsize=10, y=1.01,
    )
    for ax, nid in zip(axes, nodes):
        df_n   = df_train[df_train['node_id'] == nid].copy()
        if df_n.empty:
            continue
        benign = df_n[df_n['attack_state'] == 'none']
        ax.plot(benign['event_time'], benign['absolute_current_ma'],
                color='#2196F3', linewidth=0.4, alpha=0.7, zorder=1)
        for atk, color in ATTACK_COLORS.items():
            if atk == 'none':
                continue
            atk_df = df_n[df_n['attack_state'] == atk]
            if not atk_df.empty:
                ax.scatter(atk_df['event_time'], atk_df['absolute_current_ma'],
                           c=color, s=1.5, zorder=2, alpha=0.85, linewidths=0)
        ax.set_ylabel(f'N{nid}\n(mA)', rotation=0, labelpad=30, va='center')
        ax.set_ylim(0, 180)
        ax.yaxis.set_major_locator(MaxNLocator(3))

    axes[-1].set_xlabel('Time (UTC)')
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save('fig10_power_timeline_12h')


# ─── ML evaluation figures ─────────────────────────────────────────────────────
def generate_evaluation_figures() -> None:
    """Generates Figures 8, 9, and 11 from cached ML evaluation results."""
    print('\nLoading ML evaluation cache...')
    try:
        fpr     = np.load(os.path.join(CACHE_DIR, 'cache_ood_fpr.npy'))
        tpr     = np.load(os.path.join(CACHE_DIR, 'cache_ood_tpr.npy'))
        prec    = np.load(os.path.join(CACHE_DIR, 'cache_ood_prec.npy'))
        rec     = np.load(os.path.join(CACHE_DIR, 'cache_ood_rec.npy'))
        y_true  = np.load(os.path.join(CACHE_DIR, 'cache_ood_y_true.npy'))
        cm      = np.load(os.path.join(CACHE_DIR, 'cache_ood_cm.npy'))
        classes = np.load(os.path.join(CACHE_DIR, 'cache_ood_classes.npy'),
                          allow_pickle=True)
        fi_df   = pd.read_csv(os.path.join(CACHE_DIR, 'cache_feature_importance.csv'),
                              index_col=0)
    except FileNotFoundError:
        print(
            'Evaluation cache not found under ./cache/. '
            'Run `python eval_indist_ood.py` first.'
        )
        return

    roc_auc_val = sk_auc(fpr, tpr)
    pr_auc_val  = sk_auc(rec, prec)
    baseline_pr = y_true.mean()

    # ── Fig 8: Normalised confusion matrix ─────────────────────────────────
    print('Generating Fig 8: Confusion Matrix...')
    display_classes = [CLASS_LABELS.get(c, c) for c in classes]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
        xticklabels=display_classes, yticklabels=display_classes,
        linewidths=0.5, linecolor='white', ax=ax,
        vmin=0, vmax=1, annot_kws={'fontsize': 9},
    )
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j + 0.5, i + 0.72, f'n={cm[i, j]}',
                    ha='center', va='center', fontsize=6, color='#444')
    ax.set_xlabel('Predicted Attack Class')
    ax.set_ylabel('Ground-Truth Attack Class')
    ax.set_title('Physical Attack Attribution\nConfusion Matrix', pad=12)
    plt.tight_layout()
    _save('fig8_confusion_matrix')

    # ── Fig 9: ROC and Precision–Recall curves ─────────────────────────────
    print('Generating Fig 9: ROC & PR Curves...')
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    ax = axes[0]
    ax.plot(fpr, tpr, color='#F44336', linewidth=1.5,
            label=f'XGBoost (AUC = {roc_auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.7, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#F44336')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right', bbox_to_anchor=(0.97, 0.05), frameon=False)

    ax = axes[1]
    ax.plot(rec, prec, color='#FF9800', linewidth=1.5,
            label=f'XGBoost (AP = {pr_auc_val:.4f})')
    ax.axhline(baseline_pr, color='k', linestyle='--', linewidth=0.7,
               label=f'Chance ({baseline_pr:.4f})')
    ax.fill_between(rec, prec, alpha=0.08, color='#FF9800')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall Curve')
    ax.legend(loc='lower left', bbox_to_anchor=(0.030, 0.05), frameon=False)

    plt.tight_layout()
    _save('fig9_roc_pr_curves')

    # ── Fig 11: Feature importance ─────────────────────────────────────────
    print('Generating Fig 11: Feature Importance...')
    fi_df.index  = [FEAT_LABELS.get(i, i) for i in fi_df.index]
    fi_sorted    = fi_df['importance'].sort_values(ascending=True)

    def _feat_color(label: str) -> str:
        if 'Roll.' in label:  return '#FF9800'
        if 'Node ID' in label: return '#9E9E9E'
        return '#2196F3'

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.barh(
        fi_sorted.index, fi_sorted.values,
        color=[_feat_color(l) for l in fi_sorted.index],
        edgecolor='white', height=0.65,
    )
    for bar, val in zip(bars, fi_sorted.values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=7.5)

    ax.set_xlabel('Feature Importance (XGBoost gain)')
    ax.set_title('Feature Importance — Physical Attribution Model', pad=8)
    ax.set_xlim(0, fi_sorted.max() * 1.18)
    ax.grid(axis='y', alpha=0)

    legend_patches = [
        mpatches.Patch(color='#2196F3', label='Raw single-sample features'),
        mpatches.Patch(color='#FF9800', label='20-second rolling mean'),
        mpatches.Patch(color='#9E9E9E', label='Node identifier'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', frameon=False,
              fontsize=7.5)
    plt.tight_layout()
    _save('fig11_feature_importance')


# ─── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    print('=' * 60)
    print(' PowerTrace-IoT Figure Generator')
    print('=' * 60)

    df_train, df_test, df_all = load_datasets()
    generate_data_figures(df_train, df_test, df_all)
    generate_evaluation_figures()

    print(f'\nAll figures written to {OUT_DIR}/')
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f'  {f:45s}  {size:.1f} KB')


if __name__ == '__main__':
    main()
