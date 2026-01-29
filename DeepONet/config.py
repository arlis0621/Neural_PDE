
import os
import torch

# Data (smaller, fast-to-generate)
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_FILE = os.path.join(OUT_DIR, "antiderivative_dataset.npz")

# GRF sampling grid (finer gives better ground truth but costs time/memory)
M_FINE = 500         # reduced from 1000 -> halves sampling cost and memory
LENGTH_SCALE = 0.2
SEED = 1234

# Sensors / problem
M_SENSORS = 50       # reduce number of sensors (was 100). Good tradeoff.
NUM_U = 200          # number of distinct u functions (was 1000). total samples = NUM_U * NUM_Y_PER_U
NUM_Y_PER_U = 50     # number of y samples per u (was 100). total samples = 10k

# Model (smaller, faster)
P_LATENT = 64        # latent dimension p (was 100)
BRANCH_HIDDEN = [64, 64]
TRUNK_HIDDEN  = [64, 64]
ACTIVATION = "relu"
USE_BIAS = True

# Training (fewer steps, larger batches to utilize CPU cores)
BATCH_SIZE = 256     # larger batch reduces number of weight updates
LR = 1e-3
EPOCHS = 20          # fewer epochs => faster runs (you can increase later)
LOG_INTERVAL = 100

# Checkpoints / IO
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# DataLoader / threading (use CPU cores efficiently)
NUM_WORKERS = min(6, max(1, (os.cpu_count() or 4) - 2))  # use several workers but leave cores for the OS
PIN_MEMORY = False   # set True only if using GPU

# PyTorch threads (avoid oversubscription)
INTRA_OP_THREADS = min(8, (os.cpu_count() or 4))   # controls MKL/OpenMP threads
INTER_OP_THREADS = 2

# Device: prefer CUDA only if available and properly configured (Intel GPU typically not supported)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
