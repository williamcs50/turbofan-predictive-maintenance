"""
Turbofan synthetic sensor generator (v2).

Design goal: produce data where the degradation signal is DETECTABLE (above the
sensor noise floor) but not TRIVIAL (a single fixed threshold should not perfectly
match a learned model). Every failure mode is built to the same standard so the
model gets a fair test on all of them -- the thing v0.1 failed to do for 4 of 5.

Two knobs control the whole thing:
  TARGET_SNR_EOL  -- signal-to-noise ratio each primary channel reaches at end of life
  DISTRIBUTED     -- if True, spread signal thin across channels so no single one
                     suffices (the design that would justify a temporal model over a
                     linear baseline); if False, per-channel SNR (proven-learnable).

Learnability gate lives in scripts/run_gate.py and reads from the preprocessed
data the model sees -- not in-memory data from this generator.
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
DATA_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)          # reproducible (v0.1 set no seed)

N_ENGINES        = 50
CYCLES           = 1000
DEGRADE_FRACTION = 0.30                   # signal injected over the last 30% of life
TARGET_SNR_EOL   = 4.0                    # primary channel reaches this SNR at RUL=0
SECONDARY_SNR    = 2.0                    # secondary channels are weaker on purpose
DISTRIBUTED      = False                  # flip to True for the spread-signal design

# Sensor baselines and noise std-devs. SNR is defined per channel as delta / noise_std.
SENSORS = {
    "vibration": dict(base=0.010, noise=0.0014),   # ~ uniform(0,0.005)/sqrt(12)
    "T50":       dict(base=650.0, noise=5.0),
    "P30":       dict(base=550.0, noise=2.0),
    "Nf":        dict(base=9000.0, noise=15.0),
    "fuel_flow": dict(base=1.000, noise=0.02),
}
# mode -> {channel: (direction, role)}  direction +1 up / -1 down
MODES = {
    "HPT_wear":        {"T50": (+1, "primary"),  "P30": (-1, "secondary")},
    "Compressor_stall":{"P30": (-1, "primary"),  "Nf":  (-1, "secondary")},
    "Bearing_failure": {"vibration": (+1, "primary"), "T50": (+1, "secondary")},
    "FOD":             {"vibration": (+1, "primary"), "fuel_flow": (+1, "secondary")},
    "Overheating":     {"T50": (+1, "primary"),  "fuel_flow": (+1, "secondary")},
}


def generate_engine(engine_id, mode):
    n = CYCLES
    degrade_start = int(n * (1 - DEGRADE_FRACTION))
    rows = []
    # baseline noisy readings
    cols = {s: RNG.normal(p["base"], p["noise"], n) for s, p in SENSORS.items()}
    for ch, (direction, role) in MODES[mode].items():
        snr = (SECONDARY_SNR if role == "secondary" else TARGET_SNR_EOL)
        if DISTRIBUTED:
            snr = 1.5                       # thin, equal across affected channels
        delta_eol = snr * SENSORS[ch]["noise"] * direction
        for c in range(degrade_start, n):
            frac = (c - degrade_start) / (n - degrade_start)   # 0 -> 1 ramp
            cols[ch][c] += delta_eol * frac
    for c in range(n):
        anomaly = 1 if c >= degrade_start else 0
        row = {"engine_id": engine_id, "cycle": c, "rul": n - 1 - c,
               "failure_mode": mode, "anomaly_label": anomaly}
        for s in SENSORS:
            row[s] = cols[s][c]
        rows.append(row)
    return pd.DataFrame(rows)


def generate_dataset():
    mode_list = list(MODES)
    frames = []
    for i in range(N_ENGINES):
        mode = mode_list[i % len(mode_list)]     # balanced across modes
        frames.append(generate_engine(f"ENG_{i:03d}", mode))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate_dataset()
    out = DATA_DIR / 'synthetic_turbofan_sensors.csv'
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} rows, {df.engine_id.nunique()} engines, "
          f"{len(SENSORS)} sensors, base anomaly rate {df.anomaly_label.mean():.2f}")
    print(f"Config: SNR_EOL={TARGET_SNR_EOL}  distributed={DISTRIBUTED}")
