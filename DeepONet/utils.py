# utils.py
import time
import numpy as np
from functools import wraps
import torch

#decorator , essentially computes the time elapsed by a function
def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("%r took %f s\n" % (f.__name__, te - ts))
        return result
    return wrapper

#Compute MSE after trimming top 0.1% worst errors (same idea as original). 
def mean_squared_error_outlier(y_true, y_pred):
    
    err = np.ravel((y_true - y_pred) ** 2)
    if len(err) == 0:
        return 0.0
    cutoff = max(1, len(err) // 1000)
    err_sorted = np.sort(err)[:-cutoff] if cutoff < len(err) else err
    return float(np.mean(err_sorted))

#Why do we need to sort the error?
#Ans : Sorting the error allows us to identify and exclude the worst-performing predictions (outliers) from the mean squared error calculation.
# By trimming the top 0.1% of errors, we can get a more robust estimate of the model's performance that is not overly influenced by a small number of extreme errors. 




#Evaluate model on (sensor_values, t) -> y dataset (numpy arrays).dataset: tuple (X_sensor, X_t, y_true) all numpy arrays
def safe_test_torch(model, dataset, device="cpu", batch_size=65535):
    
    
    X_sensor, X_t, y_true = dataset
    n = X_sensor.shape[0]
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xs = torch.tensor(X_sensor[i : i + batch_size], dtype=torch.float32, device=device)
            xt = torch.tensor(X_t[i : i + batch_size], dtype=torch.float32, device=device)
            ypred = model.predict_from_parts(xs, xt)  # returns tensor shape (B,1)
            preds.append(ypred.cpu().numpy())
    ypred = np.vstack(preds)
    mse = float(np.mean((y_true - ypred) ** 2))
    mse_no_outliers = mean_squared_error_outlier(y_true, ypred)
    print("Test MSE:", mse)
    print("Test MSE w/o outliers:", mse_no_outliers)
    return mse, mse_no_outliers
