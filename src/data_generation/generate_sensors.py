import pandas as pd

import numpy as np

import random

from datetime import datetime, timedelta

# Define failure modes and their effects

failure_modes = {

    'High-pressure turbine wear': {'vib_increase': 0.002, 'temp_increase': 0.5, 'start_degrade': 50},

    'Compressor stall': {'press_drop': 0.1, 'start_degrade': 30},

    'Main bearing failure': {'vib_increase': 0.003, 'start_degrade': 40},

    'Foreign object damage': {'abrupt_vib_spike': 0.1, 'start_degrade': 20},  # Sudden

    'Overheating': {'temp_increase': 1.0, 'start_degrade': 60}

}

def generate_engine_data(engine_id, num_cycles, failure_cycle, failure_mode):

    data = []

    base_ts = datetime(2025, 11, 1, 8, 0) + timedelta(days=random.randint(0, 365))  # Random start

    rul = num_cycles

    params = failure_modes[failure_mode]

    degrade_start = num_cycles - params['start_degrade']

    

    for cycle in range(1, num_cycles + 1):

        # Base values with noise

        setting_1 = 0.45 + random.uniform(-0.05, 0.05)

        setting_2 = 0.72 + random.uniform(-0.05, 0.05)

        T2 = 518.67 + np.random.normal(0, 1)

        T24 = 642 + np.random.normal(0, 2)

        T30 = 1589 + np.random.normal(0, 5)

        T50 = 1400 + np.random.normal(0, 5)

        P15 = 14.62 + np.random.normal(0, 0.1)

        P30 = 554 + np.random.normal(0, 2)

        nf = 2388 + np.random.normal(0, 5)

        nc = 9065 + np.random.normal(0, 10)

        epr = 1.3 + np.random.normal(0, 0.01)

        ps30 = 47.47 + np.random.normal(0, 0.2)

        farb = 0.84 + np.random.normal(0, 0.001)

        vibration = 0.01 + random.uniform(0, 0.005)

        

        # Apply degradation

        if cycle > degrade_start:

            deg_factor = (cycle - degrade_start) / params['start_degrade']

            if 'vib_increase' in params:

                vibration += params['vib_increase'] * deg_factor ** 2  # Quadratic wear

            if 'temp_increase' in params:

                T50 += params['temp_increase'] * deg_factor

            if 'press_drop' in params:

                P30 -= params['press_drop'] * deg_factor

            if 'abrupt_vib_spike' in params and cycle == failure_cycle - 10:

                vibration += params['abrupt_vib_spike']  # Sudden event

        

        anomaly = 1 if vibration > 0.05 or T50 > 1420 else 0  # Thresholds based on real specs

        failed = 1 if cycle == failure_cycle else 0

        ts = base_ts + timedelta(minutes=10 * (cycle - 1))

        

        data.append({

            'engine_id': engine_id,

            'cycle': cycle,

            'timestamp': ts.strftime('%Y-%m-%d %H:%M'),

            'setting_1': setting_1, 'setting_2': setting_2,

            'T2_total_temp': T2, 'T24_total_temp': T24, 'T30_total_temp': T30, 'T50_total_temp': T50,

            'P15_static_pressure': P15, 'P30_total_pressure': P30,

            'nf_fan_speed': nf, 'nc_core_speed': nc, 'epr_engine_pressure_ratio': epr,

            'ps30_static_pressure': ps30, 'farb_fuel_air_ratio_burner': farb, 'vibration': vibration,

            'anomaly_label': anomaly, 'failure_label': failed, 'rul': rul, 'failure_mode': failure_mode if failed else ''

        })

        rul -= 1

    

    return pd.DataFrame(data)

# Generate for 500 engines

all_data = []

for i in range(1, 501):

    engine_id = f'ENG_{i:03d}'

    num_cycles = random.randint(100, 300)

    failure_cycle = num_cycles if random.random() > 0.1 else random.randint(50, num_cycles)  # 90% fail at end

    failure_mode = random.choice(list(failure_modes.keys()))

    all_data.append(generate_engine_data(engine_id, num_cycles, failure_cycle, failure_mode))

df_sensors = pd.concat(all_data)
df_sensors = df_sensors.sort_values(['engine_id', 'cycle'])
df_sensors.to_csv('synthetic_turbofan_sensors.csv', index=False)

print(f"Generated {len(df_sensors)} rows for {len(all_data)} engines!")
