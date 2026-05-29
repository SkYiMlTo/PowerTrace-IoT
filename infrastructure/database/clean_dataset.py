#!/usr/bin/env python3
"""
clean_dataset.py
================
Post-collection data-cleaning utility for the PowerTrace-IoT training dataset.

Applies two-pass physical-constraint imputation:
  • Upper bound: rows where absolute_current_ma > 250 mA or power_mw > 1250 mW
    are physically impossible for the ESP32 hardware and are treated as ADC
    glitches.
  • Lower bound: rows where absolute_current_ma < 40 mA are also physically
    impossible (a powered-on ESP32 always draws at least ~40 mA at idle) and
    are treated as transient measurement artefacts (e.g. I²C bus contention
    during the ADC read).

Both sets of anomalous samples are masked to NaN and then imputed with a
per-node rolling median (window = 10 samples ≈ 10 s).  Forward-fill then
back-fill resolves any remaining boundary NaNs.

Usage (run from the directory containing ``raw_power_dump.csv``):
    python clean_dataset.py

Input  : raw_power_dump.csv
Output : power_readings_train.csv   (drop-in replacement for the raw dump)
"""

import pandas as pd
import numpy as np

# ─── Load ──────────────────────────────────────────────────────────────────────
print("Loading raw dataset...")
df = pd.read_csv("raw_power_dump.csv", parse_dates=["event_time"])
print(f"  Original size: {len(df):,} rows")

# ─── Columns that require physical-constraint imputation ──────────────────────
COLS_TO_IMPUTE = [
    "sensor_voltage_mean_v",
    "sensor_voltage_peak_v",
    "absolute_current_ma",
    "variance_current_ma",
    "peak_current_ma",
    "current_variance_sigma_ma",
    "power_mw",
]

# ─── Define anomaly condition ─────────────────────────────────────────────────
# Upper bound: physically impossible for the ESP32 hardware (max ~200 mA).
# A 25 % tolerance (250 mA / 1 250 mW) ensures only true ADC glitches are
# flagged, not legitimate Exhaustion-attack peaks.
#
# Lower bound: a powered-on ESP32 cannot draw less than ~40 mA.
# Values below this threshold are transient ADC artefacts.
UPPER = (df["absolute_current_ma"] > 250) | (df["power_mw"] > 1250)
LOWER = df["absolute_current_ma"] < 40
condition = UPPER | LOWER

upper_count = UPPER.sum()
lower_count = LOWER.sum()
total_anomalies = condition.sum()
print(f"\n  Anomalies detected:")
print(f"    Upper-bound (>250 mA or >1250 mW) : {upper_count:,} rows")
print(f"    Lower-bound (<40 mA)               : {lower_count:,} rows")
print(f"    Total                              : {total_anomalies:,} rows "
      f"({100 * total_anomalies / len(df):.3f}% of dataset)")

# ─── Mask anomalies as NaN ────────────────────────────────────────────────────
for col in COLS_TO_IMPUTE:
    df.loc[condition, col] = np.nan

# ─── Impute with per-node rolling median ─────────────────────────────────────
# Grouped by node_id to prevent values bleeding between devices.
# Window = 10 samples ≈ 10 s at the 1-sample-per-second collection rate.
print("\n  Imputing anomalies with per-node rolling median (window=10)...")
for col in COLS_TO_IMPUTE:
    df[col] = df.groupby("node_id")[col].transform(
        lambda x: x.fillna(
            x.rolling(window=10, min_periods=1, center=True).median()
        )
    )
    # Resolve any boundary NaNs that the rolling window could not fill.
    df[col] = df.groupby("node_id")[col].transform(
        lambda x: x.ffill().bfill()
    )

# ─── Verify ───────────────────────────────────────────────────────────────────
remaining_nans = df[COLS_TO_IMPUTE].isnull().sum().sum()
if remaining_nans > 0:
    print(f"\n  WARNING: {remaining_nans} NaN(s) could not be imputed.")
else:
    print("  All anomalies successfully imputed.")

# ─── Save ─────────────────────────────────────────────────────────────────────
out_path = "power_readings_train.csv"
print(f"\n  Saving cleaned dataset → {out_path}")
df.to_csv(out_path, index=False)
print(f"  Done. {len(df):,} rows written.")
