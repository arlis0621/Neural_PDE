# src/test_env.py
import numpy as np
import torch

def main():
    print("numpy", np.__version__)
    print("torch", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    # small dummy arrays to check numpy/pytorch shapes
    X = np.random.randn(4, 64)   # pretend: batch x grid
    import math
    print("X shape:", X.shape)
    t = torch.from_numpy(X).float()
    print("torch tensor shape:", t.shape)
    if torch.cuda.is_available():
        t = t.cuda()
        print("tensor moved to GPU")

if __name__ == "__main__":
    main()
