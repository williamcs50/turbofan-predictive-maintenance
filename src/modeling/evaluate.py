import json
import torch
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import numpy as np
from train_transformer import EngineDataset, TransformerModel
from torch.utils.data import DataLoader
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'

def main():
    # Load test loader
    test_dataset = EngineDataset(DATA_DIR / 'test_sensors.csv')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
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
            anom_p = (probs >= 0.22).astype(int)
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
    cost = 50 * fn + fp

    print("Transformer Test Metrics (t*=0.22, r=50):")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    print(f"  FN:          {fn}  FP: {fp}  total_cost: {cost}")
    print(f"  RUL RMSE:    {rmse:.2f} cycles")

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
        'threshold': 0.22,
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