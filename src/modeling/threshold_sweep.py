import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from train_transformer import EngineDataset, TransformerModel

DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'

COST_FN = 50
COST_FP = 1
VAL_FRAC = 0.2


def build_val_split():
    df = pd.read_csv(DATA_DIR / 'train_sensors.csv')
    engines = sorted(df['engine_id'].unique())
    n_val = max(1, int(len(engines) * VAL_FRAC))
    val_engines = list(engines[-n_val:])
    val_df = df[df['engine_id'].isin(val_engines)]
    val_path = DATA_DIR / 'val_sensors.csv'
    val_df.to_csv(val_path, index=False)
    return val_path, val_engines


def get_probs(model, loader):
    probs, labels = [], []
    with torch.no_grad():
        for seq, anom, _ in loader:
            logits, _ = model(seq)
            p = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            labels.extend(anom.cpu().numpy())
    return np.array(probs), np.array(labels)


def run_sweep(probs, labels, thresholds):
    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        cost = COST_FN * fn + COST_FP * fp
        results.append({
            'threshold':  round(float(t), 2),
            'precision':  round(prec, 4),
            'recall':     round(rec, 4),
            'f1':         round(f1, 4),
            'FN':         fn,
            'FP':         fp,
            'total_cost': cost,
        })
    return results


def main():
    val_path, val_engines = build_val_split()
    print(f"Val split: {len(val_engines)} engines ({val_engines})")

    val_dataset = EngineDataset(val_path)
    val_loader  = DataLoader(val_dataset, batch_size=64, shuffle=False)

    all_labels  = [val_dataset[i][1].item() for i in range(len(val_dataset))]
    n_pos       = sum(all_labels)
    n_total     = len(all_labels)
    prevalence  = n_pos / n_total
    print(f"Val window prevalence: {n_pos}/{n_total} = {prevalence:.4f}")
    print(f"Cost ratio r = {COST_FN} (FN) : {COST_FP} (FP)")
    print(
        "  Rationale: a missed failure costs ~50x a false alarm, a conservative lower bound "
        "on fully-loaded turbofan miss cost (repairs, AOG/revenue loss, logistics, secondary "
        "damage) versus the cost of an unnecessary borescope inspection or targeted maintenance check."
    )

    model_path = MODELS_DIR / 'transformer_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Run train_transformer.py first.")
    model = TransformerModel(input_dim=val_dataset.num_features)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("Model loaded.")

    probs, labels = get_probs(model, val_loader)

    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    results    = run_sweep(probs, labels, thresholds)

    t_star_row = min(results, key=lambda r: r['total_cost'])

    # Find row closest to default 0.5
    default_row = min(results, key=lambda r: abs(r['threshold'] - 0.50))

    header = f"{'threshold':>9} | {'precision':>9} | {'recall':>7} | {'f1':>6} | {'FN':>5} | {'FP':>5} | {'total_cost':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        marker = ""
        if r['threshold'] == t_star_row['threshold']:
            marker = "  <-- t* (argmin cost)"
        elif r['threshold'] == default_row['threshold']:
            marker = "  <-- default 0.5"
        print(
            f"{r['threshold']:>9.2f} | {r['precision']:>9.4f} | {r['recall']:>7.4f} | "
            f"{r['f1']:>6.4f} | {r['FN']:>5} | {r['FP']:>5} | {r['total_cost']:>10}{marker}"
        )

    print(f"\nt* = {t_star_row['threshold']:.2f}  (argmin 50·FN + FP on val)")
    print(f"  precision={t_star_row['precision']:.4f}  recall={t_star_row['recall']:.4f}  "
          f"F1={t_star_row['f1']:.4f}  FN={t_star_row['FN']}  FP={t_star_row['FP']}  "
          f"total_cost={t_star_row['total_cost']}")
    print(f"\nDefault t=0.50:")
    print(f"  precision={default_row['precision']:.4f}  recall={default_row['recall']:.4f}  "
          f"F1={default_row['f1']:.4f}  FN={default_row['FN']}  FP={default_row['FP']}  "
          f"total_cost={default_row['total_cost']}")

    out = {
        'cost_ratio': {'FN': COST_FN, 'FP': COST_FP},
        'val_prevalence': round(prevalence, 4),
        'val_engines': val_engines,
        't_star': t_star_row,
        'default_05': default_row,
        'sweep': results,
    }
    results_path = MODELS_DIR / 'sweep_results.json'
    with open(results_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSweep results saved to {results_path}")


if __name__ == '__main__':
    main()
