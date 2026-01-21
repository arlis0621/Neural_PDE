# src/train.py
import numpy as np

def main():
    d = np.load('data/raw/toy_burgers.npz')
    F, U = d['F'], d['U']
    print("Loaded F shape", F.shape, "U shape", U.shape)

if __name__ == "__main__":
    main()
