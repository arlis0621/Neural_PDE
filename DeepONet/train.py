# train.py
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import config

torch.set_num_threads(config.INTRA_OP_THREADS)
torch.set_num_interop_threads(config.INTER_OP_THREADS)
from dataset import get_loaders
from model import DeepONet
from utils import set_seed
import os

def evaluate(model, loader, device):
    model.eval()
    mse = 0.0
    n = 0
    with torch.no_grad():
        for u, y, t in loader:
            u = u.to(device)
            y = y.to(device)
            t = t.to(device)
            pred = model(u, y)
            mse += torch.sum((pred - t)**2).item()
            n += t.numel()
    return mse / n

def train():
    set_seed(config.SEED)
    device = config.DEVICE
    train_loader, test_loader = get_loaders(config.DATA_FILE, config.BATCH_SIZE)
    model = DeepONet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.MSELoss(reduction='mean')

    best_test = 1e12
    step = 0
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for u, y, t in pbar:
            u = u.to(device)
            y = y.to(device)
            t = t.to(device)
            pred = model(u, y)
            loss = criterion(pred, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % config.LOG_INTERVAL == 0:
                # evaluate on test set
                test_mse = evaluate(model, test_loader, device)
                pbar.set_postfix({'step': step, 'train_loss': loss.item(), 'test_mse': test_mse})
                # checkpoint
                if test_mse < best_test:
                    best_test = test_mse
                    ckpt = os.path.join(config.CHECKPOINT_DIR, f"best_model.pt")
                    torch.save({'model_state': model.state_dict(),
                                'optimizer_state': optimizer.state_dict(),
                                'step': step,
                                'test_mse': test_mse}, ckpt)
        # end epoch
    total_time = time.time() - start_time
    print("Training finished in {:.2f} s. Best test MSE: {:.6e}".format(total_time, best_test))

if __name__ == "__main__":
    train()
