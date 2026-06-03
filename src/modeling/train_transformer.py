import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import os
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

class EngineDataset(Dataset):
    """Time-series dataset of sliding windows over engine cycles."""

    def __init__(self, csv_file: str, sequence_length: int = 50):
        df = pd.read_csv(csv_file)
        self.groups = df.groupby('engine_id')
        self.sequence_length = sequence_length

        sensor_cols = [
            col for col in df.columns
            if col not in [
                'engine_id', 'cycle', 'timestamp', 'anomaly_label',
                'failure_label', 'rul', 'failure_mode', 'num_records'
            ]
        ]
        self.num_features = len(sensor_cols)

        self.data = []
        for _, group in self.groups:
            sensors = group[sensor_cols].values
            anomalies = group['anomaly_label'].values
            ruls = group['rul'].values

            for i in range(len(group) - sequence_length):
                seq = sensors[i : i + sequence_length]
                label_anom = anomalies[i + sequence_length]
                label_rul = ruls[i + sequence_length]
                self.data.append((seq, label_anom, label_rul))

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, idx):

        seq, anom, rul = self.data[idx]

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(anom, dtype=torch.long), torch.tensor(rul, dtype=torch.float32)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]
    
class TransformerModel(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, num_layers=3, num_heads=4):

        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True, dropout=0.1)

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_anom = nn.Linear(hidden_dim, 2)  # Softmax for binary

        self.fc_rul = nn.Linear(hidden_dim, 1)  # Linear for regression

    

    def forward(self, x):

        x = self.embedding(x)  # (batch, seq_len, hidden)

        x = self.pos_encoder(x)

        x = self.transformer(x)

        x = x.mean(dim=1)  # Global average pool

        anom_out = self.fc_anom(x)

        rul_out = self.fc_rul(x)

        return anom_out, rul_out

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()

# Setup
def main():
    train_dataset = EngineDataset(DATA_DIR / 'train_sensors.csv')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    input_dim = train_dataset.num_features

    # Model and optimizer      

    model = TransformerModel(input_dim=input_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    criterion_class = FocalLoss(gamma=2.0)

    criterion_reg = nn.MSELoss()

    model_path = MODELS_DIR / 'transformer_model.pth'

    if not os.path.exists(model_path):
        # train the model from scratch
        print("Training the model from scratch....")

        epoch_losses = []

        for epoch in range(10):

            model.train()

            total_loss = 0

            for seq, anom, rul in train_loader:

                optimizer.zero_grad()

                anom_pred, rul_pred = model(seq)

                loss_class = criterion_class(anom_pred, anom)

                loss_reg = criterion_reg(rul_pred.squeeze(), rul / 125.0)

                loss = loss_class + 0.1 * loss_reg

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}: Avg Loss {avg_loss:.4f}")

        # Save model and loss history
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        history_path = MODELS_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({'epoch_losses': epoch_losses}, f)
        print(f"Loss history saved to {history_path}")

    else:
        print(f"Loading saved model: {model_path}")
        model.load_state_dict(torch.load(model_path))

    print("Training Complete!")

    # Test Eval
    model.eval()
    test_dataset = EngineDataset(DATA_DIR / 'test_sensors.csv')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    anom_preds, anom_true = [], []
    rul_preds, rul_true = [], []

    with torch.no_grad():
        for seq, anom, rul in test_loader:
            anom_logits, rul_pred = model(seq)
            probs = F.softmax(anom_logits, dim=1)[:, 1].cpu().numpy()
            anom_pred = (probs >= 0.22).astype(int)
            anom_preds.extend(anom_pred)
            anom_true.extend(anom.cpu().numpy())
            rul_preds.extend((rul_pred * 125.0).cpu().numpy())
            rul_true.extend(rul.cpu().numpy())

    acc = accuracy_score(anom_true, anom_preds)
    prec = precision_score(anom_true, anom_preds, zero_division=0)
    rec = recall_score(anom_true, anom_preds, zero_division=0)
    rmse = np.sqrt(mean_squared_error(rul_true, rul_preds))

    print("\nTest set performance (t*=0.22, r=50):")
    print(f" Anomaly Accuracy : {acc:.4f}")
    print(f" Precision        : {prec:.4f}")
    print(f" Recall           : {rec:.4f}")
    print(f" RUL RMSE         : {rmse:.2f} cycles")

if __name__ == '__main__':
    main()