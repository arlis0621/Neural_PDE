# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import config

class OperatorDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X_u = data['X_u']    # shape (N, m)
        self.X_y = data['X_y']    # shape (N, 1)
        self.y   = data['y']      # shape (N, 1)
        # convert to float32 if necessary
        self.X_u = self.X_u.astype(np.float32)
        self.X_y = self.X_y.astype(np.float32)
        self.y   = self.y.astype(np.float32)
        self.N = self.X_u.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        u = torch.from_numpy(self.X_u[idx])   # (m,)
        y = torch.from_numpy(self.X_y[idx])   # (1,)
        t = torch.from_numpy(self.y[idx])     # (1,)
        return u, y, t

def get_loaders(npz_path, batch_size, split_train=0.8, shuffle=True):
    ds = OperatorDataset(npz_path)
    N = len(ds)
    n_train = int(N * split_train)
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    from torch.utils.data import Subset
    train_ds = Subset(ds, train_idx)
    test_ds  = Subset(ds, test_idx)
    train_loader = DataLoader(train_ds,batch_size=batch_size,shuffle=True,drop_last=False,num_workers=config.NUM_WORKERS,pin_memory=config.PIN_MEMORY,)
    test_loader = DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=max(0, config.NUM_WORKERS // 2),pin_memory=config.PIN_MEMORY,)

    return train_loader, test_loader
