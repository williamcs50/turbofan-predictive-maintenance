import json
import torch
import torch.nn.functional as F
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import numpy as np
from train_transformer import EngineDataset, TransformerModel
from config import ANOMALY_THRESHOLD, COST_FN, COST_FP
from torch.utils.data import DataLoader
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'

def main():
    test_dataset = EngineDataset(DATA_DIR / 'test_sensors.csv')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Build per-window failure modes in the same groupby order EngineDataset uses
    test_df = pd.read_csv(DATA_DIR / 'test_sensors.csv')
    window_modes = []
    for _, group in test_df.groupby('engine_id'):
        mode = group['failure_mode'].iloc[0]
        n_windows = len(group) - test_dataset.sequence_length
        window_modes.extend([mode] * n_windows)
    window_modes = np.array(window_modes)
    input_dim = test_dataset.num_features

    # Load model
    model_path = MODELS_DIR / 'transformer_model.pth'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. " "Run train_transformer.py first.")
    
    model = TransformerModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f"Loaded model from {model_path}")

    anom_true, anom_pred = [], []
    rul_true, rul_pred = [], []

    with torch.no_grad():
        for seq, anom, rul in test_loader:
            anom_logits, rul_out = model(seq)
            probs = F.softmax(anom_logits, dim=1)[:, 1].cpu().numpy()
            anom_p = (probs >= ANOMALY_THRESHOLD).astype(int)
            anom_pred.extend(anom_p)
            anom_true.extend(anom.cpu().numpy())

            rul_p = (rul_out.squeeze(-1) * 125.0).cpu().numpy()
            rul_pred.extend(rul_p)
            rul_true.extend(rul.cpu().numpy())

    # Metrics
    anom_true_arr = np.array(anom_true)
    anom_pred_arr = np.array(anom_pred)
    acc = accuracy_score(anom_true_arr, anom_pred_arr)
    prec = precision_score(anom_true_arr, anom_pred_arr, zero_division=0)
    rec = recall_score(anom_true_arr, anom_pred_arr, zero_division=0)
    rmse = np.sqrt(mean_squared_error(rul_true, rul_pred))
    fn = int(((anom_pred_arr == 0) & (anom_true_arr == 1)).sum())
    fp = int(((anom_pred_arr == 1) & (anom_true_arr == 0)).sum())
    cost = COST_FN * fn + COST_FP * fp

    f1 = f1_score(anom_true_arr, anom_pred_arr, zero_division=0)

    print(f"Transformer Test Metrics (t*={ANOMALY_THRESHOLD}, r=50):")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  FN:          {fn}  FP: {fp}  total_cost: {cost}")
    print(f"  RUL RMSE:    {rmse:.2f} cycles")
    print(f"  Per-mode recall:")
    for mode in sorted(set(window_modes)):
        sel = window_modes == mode
        r = recall_score(anom_true_arr[sel], anom_pred_arr[sel], zero_division=0)
        print(f"    {mode:22s} recall {r:.4f}   n={sel.sum():5d}")

    predictions = {
        'anom_true': [int(x) for x in anom_true],
        'anom_pred': [int(x) for x in anom_pred],
        'rul_true': [float(x) for x in rul_true],
        'rul_pred': [float(x) for x in rul_pred],
    }
    out_path = MODELS_DIR / 'test_predictions.json'
    with open(out_path, 'w') as f:
        json.dump(predictions, f)
    print(f"Predictions saved to {out_path}")

    assets_dir = ROOT / 'assets'
    assets_dir.mkdir(exist_ok=True)
    metrics = {
        'threshold': ANOMALY_THRESHOLD,
        'cost_ratio': 50,
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'fn': fn,
        'fp': fp,
        'total_cost': cost,
        'rul_rmse': round(rmse, 2),
    }
    metrics_path = assets_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()