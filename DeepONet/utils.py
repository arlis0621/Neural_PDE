# utils.py
import numpy as np

def rbf_kernel(x, y, length_scale=0.2):
   
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    d2 = (x**2).sum(axis=1)[:, None] + (y**2).sum(axis=1)[None, :] - 2 * np.dot(x, y.T)
    return np.exp(-0.5 * d2 / (length_scale**2))


def cumtrapz(y, x):
    
    x = np.asarray(x)
    y = np.asarray(y)
    dx = np.diff(x)
    if y.ndim == 1:
        mid = 0.5 * (y[:-1] + y[1:])
        c = np.concatenate(([0.0], np.cumsum(mid * dx)))
        return c
    else:
        mid = 0.5 * (y[:, :-1] + y[:, 1:])
        c = np.hstack((np.zeros((y.shape[0], 1)), np.cumsum(mid * dx, axis=1)))
        return c


def set_seed(seed=0):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
