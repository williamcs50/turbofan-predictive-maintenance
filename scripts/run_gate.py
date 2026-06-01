"""
Learnability gate — reads from preprocessed train/test CSVs (same data path
the Transformer sees). Fits a plain logistic regression on windowed signal
channels and prints per-mode recall. If a mode clears 0.76, its signal survives
the full preprocessing pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'

SIGNAL_CHANNELS = ["vibration", "T50", "P30", "Nf", "fuel_flow"]
WINDOW = 50
RECALL_BAR = 0.76


def window_features(df, engines):
    """Mean of each signal channel over a WINDOW-cycle window; label = anomaly at window end."""
    X, y, modes = [], [], []
    for eng in engines:
        g = df[df.engine_id == eng].sort_values("cycle").reset_index(drop=True)
        sig = g[SIGNAL_CHANNELS].values
        anom = g["anomaly_label"].values
        mode = g["failure_mode"].iloc[0]
        for i in range(len(g) - WINDOW):
            X.append(sig[i:i + WINDOW].mean(axis=0))
            y.append(anom[i + WINDOW])
            modes.append(mode)
    return np.array(X), np.array(y), np.array(modes)


def run_gate():
    train_df = pd.read_csv(DATA_DIR / 'train_sensors.csv')
    test_df  = pd.read_csv(DATA_DIR / 'test_sensors.csv')

    Xtr, ytr, _   = window_features(train_df, train_df.engine_id.unique())
    Xte, yte, mte = window_features(test_df,  test_df.engine_id.unique())

    # Z-score for logistic regression (data is already MinMaxScaled by preprocess.py)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr_z, Xte_z = (Xtr - mu) / sd, (Xte - mu) / sd

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr_z, ytr)
    pred = clf.predict(Xte_z)

    print(f"  base rate (anomaly fraction): {yte.mean():.2f}")
    print(f"  overall  precision {precision_score(yte, pred):.2f}  "
          f"recall {recall_score(yte, pred):.2f}  f1 {f1_score(yte, pred):.2f}")
    print("  per-mode recall:")
    for m in sorted(set(mte)):
        sel = mte == m
        r = recall_score(yte[sel], pred[sel], zero_division=0)
        flag = "PASS" if r >= RECALL_BAR else "fail"
        print(f"    {m:22s} recall {r:.2f}   n={sel.sum():5d}  [{flag}]")

    best = 0.0
    for j in range(Xte_z.shape[1]):
        thr = np.quantile(Xtr_z[ytr == 0, j], 0.90)
        naive = (Xte_z[:, j] > thr).astype(int)
        best = max(best, f1_score(yte, naive, zero_division=0))
    print(f"  best single-threshold f1 (naive baseline): {best:.2f}")


if __name__ == "__main__":
    run_gate()
