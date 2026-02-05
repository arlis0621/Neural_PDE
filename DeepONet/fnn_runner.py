# fnn_runner.py

# FNN runner:
# loads dataset from artifacts/antideriv_dataset.npz
#trains FNN on same dataset and saves outputs
#produces comparison plots and metrics saved to outputs/fnn_run_<ts>/

import os
import json
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

from fnn import FNN
from models import DeepONet  # to load DeepONet best model for comparison
from utils import mean_squared_error_outlier

#-- helpers ---

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_artifact_dataset(artifact_path):
    data = np.load(artifact_path)
    X_train_sensors = data["X_train_sensors"]
    X_train_tf = data["X_train_tf"]
    y_train = data["y_train"]
    X_test_sensors = data["X_test_sensors"]
    X_test_tf = data["X_test_tf"]
    y_test = data["y_test"]
    return X_train_sensors, X_train_tf, y_train, X_test_sensors, X_test_tf, y_test

def standardize_train_test(X_train, X_test, eps=1e-9):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + eps
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std
    return X_train_n, X_test_n, mean, std

def evaluate_numpy_model(model, X_sensor, X_tf, y_true, device="cpu", batch_size=65536, is_fnn=True):
    model.eval()
    n = X_sensor.shape[0]
    preds = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.tensor(X_sensor[i:i+batch_size], dtype=torch.float32, device=device)
            xt = torch.tensor(X_tf[i:i+batch_size], dtype=torch.float32, device=device)
            if is_fnn:
                xin = torch.cat([xb, xt], dim=1)
                yp = model(xin)
            else:
                yp = model.predict_from_parts(xb, xt)
            preds.append(yp.cpu().numpy())
    ypred = np.vstack(preds)
    mse = float(np.mean((y_true - ypred)**2))
    mse_trim = mean_squared_error_outlier(y_true, ypred)
    return mse, mse_trim, ypred

# main runner 

def train_fnn_and_compare(artifact_dataset_path,
deeponet_run_meta=None,
output_root="outputs",
device="cpu",
hidden=(256,256),
dropout=0.0,
batch_size=128,
lr=1e-3,
epochs=2000,
seed=12345):
    ensure_dir(output_root)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"fnn_run_{now}")
    ensure_dir(run_dir)

    # load dataset
    X_train_sensors, X_train_tf, y_train, X_test_sensors, X_test_tf, y_test = load_artifact_dataset(artifact_dataset_path)

    # optionally load deeponet model and compute its predictions on test set
    deeponet_pred = None
    deeponet_metrics = None
    if deeponet_run_meta is not None and os.path.exists(deeponet_run_meta):
        with open(deeponet_run_meta, "r") as fh:
            meta = json.load(fh)
        deeponet_model_path = meta.get("deeponet_best_model")
        # instantiate model with saved params -- assume meta contains m, dim_x, p, branch_hidden, trunk_hidden
        m = meta.get("m", X_train_sensors.shape[1])
        dim_x = meta.get("dim_x", 1)
        p = meta.get("p", 100)
        branch_hidden = meta.get("branch_hidden", [128,128])
        trunk_hidden = meta.get("trunk_hidden", [128,128])
        device_t = torch.device("cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu")
        deeponet = DeepONet(m=m, dim_x=dim_x, branch_hidden=branch_hidden, trunk_hidden=trunk_hidden, p=p, use_bias=True)
        deeponet.to(device_t)
        deeponet.load_state_dict(torch.load(deeponet_model_path, map_location=device_t))
        # compute predictions on test set
        mse, mse_trim, ypred = evaluate_numpy_model(deeponet, X_test_sensors, X_test_tf, y_test, device=device_t, batch_size=65536, is_fnn=False)
        deeponet_pred = ypred
        deeponet_metrics = {"test_mse": mse, "test_mse_trim": mse_trim}
        np.savez_compressed(os.path.join(run_dir, "deeponet_test_preds.npz"), y_test=y_test, ypred=ypred)
        print("Loaded DeepONet and computed test predictions. MSE:", mse)

    # Train FNN on same dataset
    # input = concat(sensors, tf) => shape (N, m+1)
    X_train_in = np.concatenate([X_train_sensors, X_train_tf], axis=1)
    X_test_in = np.concatenate([X_test_sensors, X_test_tf], axis=1)

    # normalize inputs and outputs (use training set stats)
    X_train_n, X_test_n, Xin_mean, Xin_std = standardize_train_test(X_train_in, X_test_in)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-9
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    # build dataloaders
    Xin_train_tensor = torch.tensor(X_train_n, dtype=torch.float32)
    Y_train_tensor = torch.tensor(y_train_n, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(Xin_train_tensor, Y_train_tensor), batch_size=batch_size, shuffle=True)

    # model
    input_dim = X_train_n.shape[1]
    model = FNN(input_dim=input_dim, hidden=hidden, dropout=dropout)
    device_t = torch.device("cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_epoch = -1
    history = {"epoch": [], "train_loss": [], "val_mse": [], "val_mse_trim": []}

    for epoch in range(1, epochs+1):
        model.train()
        tot_loss = 0.0
        nseen = 0
        for xb, yb in train_loader:
            xb = xb.to(device_t)
            yb = yb.to(device_t)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.item()) * xb.shape[0]
            nseen += xb.shape[0]
        tot_loss /= max(1, nseen)

        # validate on test set (normalized)
        model.eval()
        with torch.no_grad():
            Xtest_tensor = torch.tensor(X_test_n, dtype=torch.float32, device=device_t)
            ypred_n = model(Xtest_tensor).cpu().numpy()
        val_mse_n = float(np.mean((y_test_n - ypred_n)**2))
        val_mse = val_mse_n * (y_std.item() ** 2)
        val_mse_trim = mean_squared_error_outlier(y_test, ypred_n * y_std + y_mean)

        history["epoch"].append(epoch)
        history["train_loss"].append(tot_loss)
        history["val_mse"].append(val_mse)
        history["val_mse_trim"].append(val_mse_trim)

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(run_dir, "fnn_best.pth"))

        if epoch % max(1, epochs//10) == 0 or epoch <= 3:
            print(f"[FNN] epoch {epoch} train_loss={tot_loss:.4e} val_mse={val_mse:.4e}")

    # finished training. compute final preds (original scale)
    model.load_state_dict(torch.load(os.path.join(run_dir, "fnn_best.pth"), map_location=device_t))
    Xtest_tensor = torch.tensor(X_test_n, dtype=torch.float32, device=device_t)
    with torch.no_grad():
        ypred_n = model(Xtest_tensor).cpu().numpy()
    ypred = ypred_n * y_std + y_mean

    test_mse = float(np.mean((y_test - ypred)**2))
    test_mse_trim = mean_squared_error_outlier(y_test, ypred)

    # save outputs
    np.savez_compressed(os.path.join(run_dir, "fnn_outputs.npz"),
                        X_test_sensors=X_test_sensors, X_test_tf=X_test_tf, y_test=y_test, ypred_fnn=ypred)
    if deeponet_pred is not None:
        # save both predictions for direct comparison
        np.savez_compressed(os.path.join(run_dir, "comparison_preds.npz"),
                            y_test=y_test, ypred_deeponet=deeponet_pred, ypred_fnn=ypred)

    metrics = {
        "fnn_test_mse": float(test_mse),
        "fnn_test_mse_trim": float(test_mse_trim),
        "fnn_best_epoch": int(best_epoch)
    }
    if deeponet_metrics is not None:
        metrics["deeponet_test_mse"] = float(deeponet_metrics["test_mse"])
        metrics["deeponet_test_mse_trim"] = float(deeponet_metrics["test_mse_trim"])

    pd.DataFrame([metrics]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # plots: parity comparison and histogram of errors
    # parity: deeponet vs fnn (if both exist)
    fig = plt.figure(figsize=(6,6))
    plt.scatter(y_test.ravel(), ypred.ravel(), s=4, alpha=0.6, label="FNN")
    if deeponet_pred is not None:
        plt.scatter(y_test.ravel(), deeponet_pred.ravel(), s=4, alpha=0.6, label="DeepONet")
    mn = min(y_test.min(), ypred.min(), (deeponet_pred.min() if deeponet_pred is not None else np.inf))
    mx = max(y_test.max(), ypred.max(), (deeponet_pred.max() if deeponet_pred is not None else -np.inf))
    plt.plot([mn, mx], [mn, mx], "k--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.legend()
    plt.title("Parity plot: test set")
    plt.savefig(os.path.join(run_dir, "parity_compare.png"), dpi=200)
    plt.close(fig)

    # error histograms
    fig = plt.figure(figsize=(8,4))
    err_fnn = (y_test - ypred).ravel()
    plt.hist(err_fnn, bins=80, alpha=0.7, label="FNN error")
    if deeponet_pred is not None:
        err_deeponet = (y_test - deeponet_pred).ravel()
        plt.hist(err_deeponet, bins=80, alpha=0.7, label="DeepONet error")
    plt.legend()
    plt.title("Error histograms")
    plt.savefig(os.path.join(run_dir, "error_histograms.png"), dpi=200)
    plt.close(fig)

    print("FNN run saved to:", run_dir)
    return run_dir, metrics