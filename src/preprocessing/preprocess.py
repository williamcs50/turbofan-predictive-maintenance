import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'data'

# Load data
df_sensors = pd.read_csv(DATA_DIR / 'synthetic_turbofan_sensors.csv')

with open(DATA_DIR / 'synthetic_maintenance_records.json', 'r') as f:
    records = json.load(f)

df_records = pd.DataFrame(records)

# Define sensor columns
sensor_cols = [
    col for col in df_sensors.columns
    if col not in [
        'engine_id', 'cycle', 'timestamp', 'anomaly_label',
        'failure_label', 'rul', 'failure_mode'
    ]
]

# Engine-level train/test split
train_engines, test_engines = train_test_split(
    df_sensors['engine_id'].unique(),
    test_size = 0.2,
    random_state=42,
)

train_df = df_sensors[df_sensors['engine_id'].isin(train_engines)].copy()
test_df = df_sensors[df_sensors['engine_id'].isin(test_engines)].copy()

# Normalize sensors
scaler = MinMaxScaler()
train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])

train_df = train_df.merge(
    df_records.groupby('engine_id').size().reset_index(name='num_records'),
    on='engine_id', how='left'
)

test_df = test_df.merge(
    df_records.groupby('engine_id').size().reset_index(name='num_records'),
    on='engine_id', how='left'
)

print(f"Original RUL range: {train_df['rul'].min():.1f} - {train_df['rul'].max():.1f}")
train_df['rul'] = train_df['rul'].clip(upper=125)
test_df['rul'] = test_df['rul'].clip(upper=125)
print(f"Capped RUL range:   {train_df['rul'].min():.1f} - {train_df['rul'].max():.1f}")

train_df.to_csv(DATA_DIR / 'train_sensors.csv', index=False)
test_df.to_csv(DATA_DIR / 'test_sensors.csv', index=False)

print("Preprocessing complete.")