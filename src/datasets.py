# src/datasets.py
import numpy as np
import os

def make_toy_dataset(path='data/raw/toy_burgers.npz', M=128, N=64):
    xs = np.linspace(0,1,N)
    F = []
    U = []
    for i in range(M):
        # simple random initial condition: sum of a few sines
        a = np.random.randn(4)
        freqs = np.array([1,2,4,8])
        f = sum(a[j]*np.sin(2*np.pi*freqs[j]*xs) for j in range(len(a)))
        # toy 'solution' = smoothed version (convolution) to mimic PDE effect
        u = np.convolve(f, np.ones(5)/5, mode='same')
        F.append(f)
        U.append(u)
    F = np.array(F, dtype=np.float32)
    U = np.array(U, dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, F=F, U=U, x=xs)
    print("Saved toy dataset:", path, "shapes", F.shape, U.shape)

if __name__ == "__main__":
    make_toy_dataset()
