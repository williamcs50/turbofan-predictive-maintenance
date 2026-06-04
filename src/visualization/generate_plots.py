import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / 'models'
ASSETS_DIR = ROOT / 'assets'
ASSETS_DIR.mkdir(exist_ok=True)

with open(MODELS_DIR / 'test_predictions.json') as f:
    preds = json.load(f)

anom_true = np.array(preds['anom_true'])
anom_pred = np.array(preds['anom_pred'])
rul_true = np.array(preds['rul_true'])
rul_pred = np.array(preds['rul_pred'])


def plot_confusion_matrix():
    cm = confusion_matrix(anom_true, anom_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly'],
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Anomaly Detection — Confusion Matrix')
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'confusion_matrix.png', dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")


def plot_rul_scatter():
    plt.figure(figsize=(7, 7))
    plt.scatter(rul_true, rul_pred, alpha=0.3, s=10, color='steelblue')
    max_val = max(rul_true.max(), rul_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect prediction')
    plt.xlabel('True RUL (cycles)')
    plt.ylabel('Predicted RUL (cycles)')
    plt.title('RUL Prediction — Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'rul_scatter.png', dpi=150)
    plt.close()
    print("Saved rul_scatter.png")


def plot_precision_recall_curve():
    sweep_path = MODELS_DIR / 'sweep_results.json'
    if not sweep_path.exists():
        print("Skipping precision-recall curve — sweep_results.json not found. Run: PYTHONPATH=src python src/modeling/threshold_sweep.py")
        return
    with open(sweep_path) as f:
        data = json.load(f)

    sweep   = data['sweep']
    t_star  = data['t_star']
    default = data['default_05']

    recall    = [r['recall']    for r in sweep]
    precision = [r['precision'] for r in sweep]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='steelblue', linewidth=2, label='PR curve (val)')

    ax.scatter(
        [t_star['recall']], [t_star['precision']],
        color='red', s=120, zorder=5,
        label=(
            f"t*={t_star['threshold']:.2f}  "
            f"P={t_star['precision']:.2f}  R={t_star['recall']:.2f}  "
            f"cost={t_star['total_cost']}"
        ),
    )

    ax.scatter(
        [default['recall']], [default['precision']],
        color='orange', s=100, marker='D', zorder=5,
        label=(
            f"default t=0.50  "
            f"P={default['precision']:.2f}  R={default['recall']:.2f}  "
            f"cost={default['total_cost']}"
        ),
    )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve — Validation Set\n(cost ratio: 50·FN + 1·FP)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / 'precision_recall_curve.png', dpi=150)
    plt.close()
    print("Saved precision_recall_curve.png")


if __name__ == '__main__':
    plot_confusion_matrix()
    plot_rul_scatter()
    plot_precision_recall_curve()
    print("Done. PNGs saved to assets/")
