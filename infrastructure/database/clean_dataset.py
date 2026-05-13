import pandas as pd
import numpy as np

# Load the raw data
print("Loading raw dataset...")
df = pd.read_csv('raw_power_dump.csv', parse_dates=['event_time'])

print(f"Original size: {len(df)}")

# Define columns to impute
cols_to_impute = [
    'sensor_voltage_mean_v', 'sensor_voltage_peak_v', 
    'absolute_current_ma', 'variance_current_ma', 'peak_current_ma', 
    'current_variance_sigma_ma', 'power_mw'
]

# Identify anomalies based on physical constraints (max 200mA -> 1000mW)
# Some tolerance added (e.g. 250mA / 1250mW) to be safe and only clip the actual glitches
condition = (df['absolute_current_ma'] > 250) | (df['power_mw'] > 1250)
anomaly_count = condition.sum()
print(f"Found {anomaly_count} anomalous rows exceeding physical limits.")

# Mask anomalies with NaN
for col in cols_to_impute:
    df.loc[condition, col] = np.nan

# Impute using rolling median (window = 5 samples, approx 5 seconds)
# Group by node_id so we don't bleed values across devices
print("Imputing anomalies using rolling median...")
for col in cols_to_impute:
    df[col] = df.groupby('node_id')[col].transform(lambda x: x.fillna(x.rolling(window=10, min_periods=1, center=True).median()))
    
    # If any NaNs remain (e.g., at the very boundaries), use forward fill then backward fill
    df[col] = df.groupby('node_id')[col].transform(lambda x: x.ffill().bfill())

# Check if any NaNs are left
if df[cols_to_impute].isnull().any().any():
    print("WARNING: Some NaNs could not be imputed.")
else:
    print("All anomalies successfully imputed.")

# Print stats for DDoS and Exhaustion to verify
print("\n--- Variance check on cleaned data ---")
print("Power (mW) Variance:")
print(df[df['attack_state'].isin(['ddos', 'exhaustion'])].groupby('attack_state')['power_mw'].var())
print("Current (mA) Variance:")
print(df[df['attack_state'].isin(['ddos', 'exhaustion'])].groupby('attack_state')['absolute_current_ma'].var())

# Save to CSV
print("\nSaving clean dataset...")
df.to_csv('power_readings_train_clean.csv', index=False)
print("Done. Saved as power_readings_train_clean.csv")
