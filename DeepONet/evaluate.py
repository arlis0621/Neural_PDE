# evaluate.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import OperatorDataset
from model import DeepONet
import config
import os

def load_model(ckpt_path, device):
    m = DeepONet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    return m

def full_test(model, npz_path):
    ds = OperatorDataset(npz_path)
    X_u = ds.X_u
    X_y = ds.X_y
    y_true = ds.y
    device = config.DEVICE
    model.to(device)
    with torch.no_grad():
        u_t = torch.from_numpy(X_u).to(device)
        y_t = torch.from_numpy(X_y).to(device)
        preds = []
        B = 8192
        for i in range(0, len(u_t), B):
            u_batch = u_t[i:i+B]
            y_batch = y_t[i:i+B]
            preds.append(model(u_batch, y_batch).cpu().numpy())
        preds = np.vstack(preds)
    mse = np.mean((preds - y_true)**2)
    print("Full Test MSE:", mse)
    return preds

if __name__ == "__main__":
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit("No checkpoint found. Train first.")
    model = load_model(ckpt_path, config.DEVICE)
    preds = full_test(model, config.DATA_FILE)
    # Example: plot predictions vs truth for first ~100 samples
    import matplotlib.pyplot as plt
    data = np.load(config.DATA_FILE)
    X_y = data['X_y'].flatten()
    true = data['y'].flatten()
    plt.figure(figsize=(6,4))
    plt.scatter(true[:200], preds[:200], s=6, alpha=0.4)
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.title("Pred vs True (sample)")
    plt.grid(True)
    plt.show()
