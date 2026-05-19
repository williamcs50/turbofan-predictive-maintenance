import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math
import os

class EngineDataset(Dataset):
    """Time-series dataset of sliding windows over engine cycles."""

    def __init__(self, csv_file: str, sequence_length: int = 50):
        df = pd.read_csv(csv_file)
        self.groups = df.groupby('engine_id')
        self.sequence_length = sequence_length

        sensor_cols = [
            col for col in df.columns
            if any(kw in col.lower() for kw in [
                'temp', 'pressure', 'speed', 'vibration', 'setting', 'epr', 'ps30', 'farb'
            ])
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

# Setup
def main():
    train_dataset = EngineDataset('train_sensors.csv')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    input_dim = train_dataset.num_features

    # Compute class weights since anomalies are rare
    labels = [anom.item() for _, anom, _ in train_dataset]
    class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array([0, 1]),
            y=np.array(labels),
        )
    
    # Clip extreme class weights to avoid gradient instability
    class_weights = np.clip(class_weights, 0, 50)

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print(f"Class weights => Normal: {class_weights[0]:.2f}, Anomaly: {class_weights[1]:.2f}")

    # Model and optimizer

    model = TransformerModel(input_dim=input_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    criterion_class = nn.CrossEntropyLoss(weight=weights_tensor)

    criterion_reg = nn.MSELoss()

    model_path = 'transformer_model.pth'

    if not os.path.exists(model_path):
        # train the model from scratch
        print("Training the model from scratch....")

        for epoch in range(10):

            model.train()

            total_loss = 0

            for seq, anom, rul in train_loader:

                optimizer.zero_grad()

                anom_pred, rul_pred = model(seq)

                loss_class = criterion_class(anom_pred, anom)

                loss_reg = criterion_reg(rul_pred.squeeze(), rul)

                loss = loss_class + 0.001 * loss_reg

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}: Avg Loss {total_loss / len(train_loader):.4f}")

        # Save once after all epochs
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    else:
        print(f"Loading saved model: {model_path}")
        model.load_state_dict(torch.load(model_path))

    print("Training Complete!")

    # Test Eval
    model.eval()
    test_dataset = EngineDataset('test_sensors.csv')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    anom_preds, anom_true = [], []
    rul_preds, rul_true = [], []

    with torch.no_grad():
        for seq, anom, rul in test_loader:
            anom_logits, rul_pred = model(seq)
            anom_pred = torch.argmax(anom_logits, dim=1)
            anom_preds.extend(anom_pred.cpu().numpy())
            anom_true.extend(anom.cpu().numpy())
            rul_preds.extend(rul_pred.cpu().numpy())
            rul_true.extend(rul.cpu().numpy())

    acc = accuracy_score(anom_true, anom_preds)
    prec = precision_score(anom_true, anom_preds, zero_division=0)
    rec = recall_score(anom_true, anom_preds, zero_division=0)
    rmse = np.sqrt(mean_squared_error(rul_true, rul_preds))

    print("\nTest set performance:")
    print(f" Anomaly Accuracy : {acc:.4f}")
    print(f" Precision        : {prec:.4f}")
    print(f" Recall           : {rec:.4f}")
    print(f" RUL RMSE         : {rmse:.2f} cycles")

if __name__ == '__main__':
    main()