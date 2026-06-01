"""Diagnostic: plot raw vs scaled sensor traces for one engine, marking the RUL≤30 anomaly window."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'

ENGINE_ID = 'ENG_003'
SENSORS = ['vibration', 'T50_total_temp', 'P30_total_pressure']

raw_df = pd.read_csv(DATA_DIR / 'synthetic_turbofan_sensors.csv')
scaled_df = pd.read_csv(DATA_DIR / 'train_sensors.csv')

# Fall back to test set if engine is in test split
if ENGINE_ID not in scaled_df['engine_id'].values:
    scaled_df = pd.read_csv(DATA_DIR / 'test_sensors.csv')

eng_raw = raw_df[raw_df['engine_id'] == ENGINE_ID].sort_values('cycle')
eng_scaled = scaled_df[scaled_df['engine_id'] == ENGINE_ID].sort_values('cycle')

if eng_raw.empty:
    raise ValueError(f"{ENGINE_ID} not found in raw data.")
if eng_scaled.empty:
    raise ValueError(f"{ENGINE_ID} not found in train or test split — check ENGINE_ID.")

anomaly_start_cycle = eng_raw[eng_raw['anomaly_label'] == 1]['cycle'].min()
max_cycle = eng_raw['cycle'].max()
failure_mode = eng_raw['failure_mode'].dropna().iloc[-1] if eng_raw['failure_mode'].dropna().any() else 'unknown'

print(f"Engine: {ENGINE_ID} | Failure mode: {failure_mode}")
print(f"Total cycles: {max_cycle} | Anomaly window starts at cycle: {anomaly_start_cycle}")
print(f"Anomaly proportion: {eng_raw['anomaly_label'].mean():.1%}")

fig, axes = plt.subplots(len(SENSORS), 2, figsize=(14, 4 * len(SENSORS)))
fig.suptitle(f"{ENGINE_ID} — {failure_mode}\nRed shading = RUL≤30 anomaly window", fontsize=13)

for row, sensor in enumerate(SENSORS):
    for col, (df, label) in enumerate([(eng_raw, 'Raw (pre-scaling)'), (eng_scaled, 'Scaled (post-MinMaxScaler)')]):
        ax = axes[row][col]
        if sensor not in df.columns:
            ax.text(0.5, 0.5, f'{sensor}\nnot found', ha='center', va='center', transform=ax.transAxes)
            continue
        ax.plot(df['cycle'], df[sensor], linewidth=0.8, color='steelblue')
        ax.axvspan(anomaly_start_cycle, max_cycle, alpha=0.15, color='red', label='RUL≤30')
        ax.axvline(anomaly_start_cycle, color='red', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'{sensor} — {label}', fontsize=10)
        ax.set_xlabel('Cycle')
        ax.set_ylabel(sensor)

red_patch = mpatches.Patch(color='red', alpha=0.3, label='Anomaly window (RUL≤30)')
fig.legend(handles=[red_patch], loc='lower right', fontsize=10)
plt.tight_layout()

out_path = ROOT / 'assets' / 'diagnose_sensor_signal.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved to {out_path}")
plt.show()
