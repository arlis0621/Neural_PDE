# data_gen.py
import numpy as np
from scipy.linalg import cholesky
from scipy.interpolate import interp1d
from utils import rbf_kernel, cumtrapz, set_seed
import config

def sample_grf(num_u, Nfine, length_scale, x_fine):
    """Sample num_u GRF sample paths on x_fine using RBF kernel."""
    K = rbf_kernel(x_fine, x_fine, length_scale=length_scale)
    # add tiny jitter
    K += 1e-12 * np.eye(len(x_fine))
    L = cholesky(K, lower=True)
    z = np.random.randn(len(x_fine), num_u)
    samples = (L @ z).T  # shape (num_u, Nfine)
    return samples

def build_dataset():
    set_seed(config.SEED)
    # fine grid
    x_fine = np.linspace(0.0, 1.0, config.M_FINE)
    # sample functions
    print("Sampling GRF functions...")
    U_fine = sample_grf(config.NUM_U, config.M_FINE, config.LENGTH_SCALE, x_fine)  # (NUM_U, M_FINE)

    # compute cumulative integrals on fine grid (antiderivative)
    print("Computing cumulative integrals...")
    S_fine = cumtrapz(U_fine, x_fine)  # shape (NUM_U, M_FINE)

    # sensor locations (same for all u)
    sensors = np.linspace(0.0, 1.0, config.M_SENSORS)

    # sample y positions for each u (random uniform in [0,1])
    total_samples = config.NUM_U * config.NUM_Y_PER_U
    print(f"Generating {total_samples} labeled triples ({config.NUM_U} u × {config.NUM_Y_PER_U} y) ...")

    # For efficiency, precompute interpolation functions for integrals and for u on sensors
    U_sensors = np.empty((config.NUM_U, config.M_SENSORS), dtype=np.float32)
    Ys = np.empty((total_samples, 1), dtype=np.float32)
    Targets = np.empty((total_samples, 1), dtype=np.float32)

    idx = 0
    for i in range(config.NUM_U):
        u_i = U_fine[i]        # fine samples
        s_i = S_fine[i]        # cumulative integral on fine grid

        # interpolate u on sensors (branch input)
        u_interp = interp1d(x_fine, u_i, kind='cubic')
        U_sensors[i, :] = u_interp(sensors)

        # interpolation for antiderivative
        s_interp = interp1d(x_fine, s_i, kind='linear')

        # sample y's
        y_samples = np.random.rand(config.NUM_Y_PER_U, 1).astype(np.float32)  # in [0,1)
        Ys[idx: idx + config.NUM_Y_PER_U, :] = y_samples
        Targets[idx: idx + config.NUM_Y_PER_U, 0] = s_interp(y_samples.flatten())
        idx += config.NUM_Y_PER_U

    # Now tile U_sensors so that each row aligns with corresponding y
    U_sensors_tiled = np.repeat(U_sensors, config.NUM_Y_PER_U, axis=0)  # shape (total_samples, M_SENSORS)

    # Save dataset
    print("Saving dataset to", config.DATA_FILE)
    np.savez_compressed(config.DATA_FILE,
                        X_u=U_sensors_tiled.astype(np.float32),
                        X_y=Ys,
                        y=Targets,
                        sensors=sensors.astype(np.float32),
                        x_fine=x_fine.astype(np.float32))
    print("Done.")

if __name__ == "__main__":
    build_dataset()
