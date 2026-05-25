import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from modeling.train_transformer import EngineDataset, TransformerModel
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
            anom_p = torch.argmax(anom_logits, dim=1).cpu().numpy()
            anom_pred.extend(anom_p)
            anom_true.extend(anom.cpu().numpy())

            rul_p = rul_out.squeeze(-1).cpu().numpy()
            rul_pred.extend(rul_p)
            rul_true.extend(rul.cpu().numpy())

    # Metrics
    acc = accuracy_score(anom_true, anom_pred)
    prec = precision_score(anom_true, anom_pred, zero_division=0)
    rec = recall_score(anom_true, anom_pred, zero_division=0)
    rmse = np.sqrt(mean_squared_error(rul_true, rul_pred))
    
    print("Transformer Test Metrics:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {rec:.4f}")
    print(f"  RUL RMSE: {rmse:.2f} cycles")

    # Simple visualization
    plt.figure(figsize=(10,6))
    plt.plot(rul_true[:100], label='True RUL', linewidth=1.5)
    plt.plot(rul_pred[:100], label='Predicted RUL', linewidth=1.5, alpha=0.8)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Remaining Useful Life (cycles)')
    plt.title('True vs Predicted RUL (first 100 test samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()