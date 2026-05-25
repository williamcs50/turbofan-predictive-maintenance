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


if __name__ == '__main__':
    plot_confusion_matrix()
    plot_rul_scatter()
    print("Done. PNGs saved to assets/")
