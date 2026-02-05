# deeponet_pde_pytorch.py

#Modified DeepONet antiderivative runner that:
#generates dataset and saves it to artifacts/antideriv_dataset.npz
#trains DeepONet, saves best model + scalers + predictions + metrics to outputs/deeponet_run_<ts>/
#writes run_meta.json so FNN runner can reuse dataset and best model.
import os
import json
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from utils import timing, mean_squared_error_outlier
from spaces import GRF
from system import ODESystem
from models import DeepONet

# -- helpers -----

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_npz_dataset(path, X_train_sensors, X_train_tf, y_train, X_test_sensors, X_test_tf, y_test):
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path,
                        X_train_sensors=X_train_sensors,
                        X_train_tf=X_train_tf,
                        y_train=y_train,
                        X_test_sensors=X_test_sensors,
                        X_test_tf=X_test_tf,
                        y_test=y_test)

def prepare_dataloader(sensor_values, x_vals, y_vals, batch_size=128, shuffle=True):
    Xb = torch.tensor(sensor_values, dtype=torch.float32)
    Xt = torch.tensor(x_vals, dtype=torch.float32)
    Y = torch.tensor(y_vals, dtype=torch.float32)
    ds = TensorDataset(Xb, Xt, Y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader


#Evaluate model on numpy arrays and return (mse, mse_trim, ypred)
def evaluate_model_numpy(model, X_sensor, X_tf, y_true, device="cpu", batch_size=65536):
    
    model.eval()
    n = X_sensor.shape[0]
    preds = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.tensor(X_sensor[i:i+batch_size], dtype=torch.float32, device=device)
            xt = torch.tensor(X_tf[i:i+batch_size], dtype=torch.float32, device=device)
            ypred = model.predict_from_parts(xb, xt).cpu().numpy()
            preds.append(ypred)
    ypred = np.vstack(preds)
    mse = float(np.mean((y_true - ypred) ** 2))
    mse_trim = mean_squared_error_outlier(y_true, ypred)
    return mse, mse_trim, ypred

# -- main routines --



#Generate antiderivative dataset and save to artifact_path (npz).
#Returns tuple of numpy arrays as loaded from the file for immediate use.
    
@timing
def gen_and_save_antideriv_dataset(artifact_path,T=1.0,m=100,num_train=2000,num_test=2000,grf_N=100,grf_length_scale=0.2,seed=12345,force_regen=False):
    
    if (not force_regen) and os.path.exists(artifact_path):
        print("Dataset file exists, loading:", artifact_path)
        data = np.load(artifact_path)
        return (data["X_train_sensors"], data["X_train_tf"], data["y_train"],
                data["X_test_sensors"], data["X_test_tf"], data["y_test"])

    np.random.seed(seed)
    print("Generating GRF and operator data (this can take time)...")
    space = GRF(T, kernel="RBF", length_scale=grf_length_scale, N=grf_N, interp="cubic")
    s0 = [0.0]
    def g(s, u, t): 
        return u  
    # antiderivative
    system = ODESystem(g, s0, T)

    (X_train_parts, x_train), y_train = system.gen_operator_data(space, m, num_train)
    (X_test_parts, x_test), y_test = system.gen_operator_data(space, m, num_test)

    # Ensure shapes
    X_train_sensors = np.asarray(X_train_parts).reshape(num_train, m)
    X_train_tf = np.asarray(x_train).reshape(num_train, 1)
    y_train = np.asarray(y_train).reshape(num_train, 1)

    X_test_sensors = np.asarray(X_test_parts).reshape(num_test, m)
    X_test_tf = np.asarray(x_test).reshape(num_test, 1)
    y_test = np.asarray(y_test).reshape(num_test, 1)

    print("Saving dataset to", artifact_path)
    save_npz_dataset(artifact_path, X_train_sensors, X_train_tf, y_train, X_test_sensors, X_test_tf, y_test)
    return X_train_sensors, X_train_tf, y_train, X_test_sensors, X_test_tf, y_test



#Train DeepONet on dataset saved at artifact_dataset_path. Saves outputs to output_root/deeponet_run_<ts>.
#Returns path to run folder.
def train_deeponet_and_save(artifact_dataset_path,output_root="outputs",device="cpu",m=100,dim_x=1,p=100,branch_hidden=[128, 128],trunk_hidden=[128, 128],lr=1e-3,batch_size=128,epochs=5000,seed=12345):   
    ensure_dir(output_root)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"deeponet_run_{now}")
    ensure_dir(run_dir)
    model_dir = os.path.join(run_dir, "model")
    ensure_dir(model_dir)

    # load dataset
    data = np.load(artifact_dataset_path)
    X_train = data["X_train_sensors"]
    x_train = data["X_train_tf"]
    y_train = data["y_train"]
    X_test = data["X_test_sensors"]
    x_test = data["X_test_tf"]
    y_test = data["y_test"]

    # build model
    model = DeepONet(m=m, dim_x=dim_x, branch_hidden=branch_hidden, trunk_hidden=trunk_hidden, p=p, use_bias=True)
    device = torch.device(device)
    model.to(device)

    train_loader = prepare_dataloader(X_train, x_train, y_train, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_step = -1
    history = {"step": [], "train_loss": [], "val_mse": [], "val_mse_trim": []}

    total_steps = 0
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss_total = 0.0
        nseen = 0
        for xb, xt, yb in train_loader:
            xb = xb.to(device)
            xt = xt.to(device)
            yb = yb.to(device)
            pred = model(xb, xt)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_n = xb.shape[0]
            epoch_loss_total += float(loss.item()) * batch_n
            nseen += batch_n
            total_steps += 1
        epoch_loss = epoch_loss_total / max(1, nseen)

        # every epoch evaluate on test set (or do this less frequently for speed)
        val_mse, val_mse_trim, _ = evaluate_model_numpy(model, X_test, x_test, y_test, device=device)
        history["step"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_mse"].append(val_mse)
        history["val_mse_trim"].append(val_mse_trim)

        if val_mse < best_val:
            best_val = val_mse
            best_step = epoch
            best_path = os.path.join(model_dir, "deeponet_best.pth")
            torch.save(model.state_dict(), best_path)
            # save scalers (none used for DeepONet here, but keep place-holder)
            np.savez_compressed(os.path.join(model_dir, "scalers.npz"),
                                X_mean=np.array([0.0]), X_std=np.array([1.0]),
                                y_mean=np.array([0.0]), y_std=np.array([1.0]))
        if epoch % max(1, epochs // 10) == 0 or epoch <= 3:
            print(f"[Epoch {epoch}] train_loss={epoch_loss:.3e} val_mse={val_mse:.3e}")

    elapsed = time.time() - start_time

    # restore best model for final evaluation
    best_path = os.path.join(model_dir, "deeponet_best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("Loaded best model from", best_path)

    # final predictions & metrics (on test set)
    test_mse, test_mse_trim, ypred = evaluate_model_numpy(model, X_test, x_test, y_test, device=device)
    # save outputs
    np.savez_compressed(os.path.join(run_dir, "deeponet_outputs.npz"),
                        X_test_sensors=X_test, X_test_tf=x_test, y_test=y_test, ypred=ypred)
    # save history & metrics
    np.savez_compressed(os.path.join(run_dir, "history.npz"), **history)
    metrics = {"test_mse": float(test_mse), "test_mse_trim": float(test_mse_trim), "best_epoch": int(best_step), "elapsed_sec": float(elapsed)}
    with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    # save run meta so FNN runner can discover model + dataset
    run_meta = {
        "artifact_dataset": os.path.abspath(artifact_dataset_path),
        "deeponet_best_model": os.path.abspath(best_path),
        "deeponet_run_dir": os.path.abspath(run_dir),
        "m": int(m),
        "dim_x": int(dim_x),
        "p": int(p),
        "branch_hidden": branch_hidden,
        "trunk_hidden": trunk_hidden,
        "device": str(device)
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w") as fh:
        json.dump(run_meta, fh, indent=2)

    print("DeepONet run saved to:", run_dir)
    return run_dir, run_meta

# -- script entry ---

if __name__ == "__main__":
    # configure hyperparameters here
    artifact_path = os.path.join("artifacts", "antideriv_dataset.npz")
    ensure_dir("artifacts")

    # generate dataset and save it
    gen_and_save_antideriv_dataset(artifact_path,T=1.0,m=100,num_train=2000,num_test=2000,grf_N=100,
grf_length_scale=0.2,
seed=12345,
force_regen=False)

    # train and save DeepONet outputs
    run_dir, run_meta = train_deeponet_and_save(artifact_dataset_path=artifact_path,output_root="outputs",device="cuda" if torch.cuda.is_available() else "cpu",m=100,dim_x=1,p=100,branch_hidden=[128, 128],trunk_hidden=[128, 128],lr=1e-3,batch_size=128,epochs=2000,seed=12345)