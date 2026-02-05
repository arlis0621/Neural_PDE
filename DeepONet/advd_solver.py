# advd_solver.py
"""
Advection-diffusion (ADVD) solver utilities.

Solves u_t + u_x - D * u_xx = 0 with periodic boundary conditions
using a Crank-Nicolson-like implicit method in space.

Functions:
- solve_ADVD(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt, D=0.1)
    f, g are optional (not used for periodic initial V-based problems).
    V(x) initial condition callable.
Returns x, t, u (Nx x Nt)
"""
import numpy as np


def solve_ADVD(xmin, xmax, tmin, tmax, f, g, V, Nx, Nt, D=0.1):
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h
    mu = dt / (h ** 2)

    # initialize
    u = np.zeros((Nx, Nt))
    if V is None and f is not None:
        u[:, 0] = f(x)
    elif V is not None:
        u[:, 0] = V(x)
    else:
        raise ValueError("No initial profile provided (V or f).")

    # Interior indices 1..Nx-1 (we will enforce periodic BC by mapping index 0 and Nx-1)
    Nint = Nx - 1  # exclude the first index (we'll store 1..Nx-1)
    # Build matrices for CN scheme as in provided snippet
    I = np.eye(Nint)
    I1 = np.roll(I, 1, axis=0)   # shifted down
    I2 = np.roll(I, -1, axis=0)  # shifted up

    A = (1 + D * mu) * I - (lam / 4 + D * mu / 2) * I1 + (lam / 4 - D * mu / 2) * I2
    B = 2 * I - A
    # Pre-factorize solve(A, B @ u)
    C = np.linalg.solve(A, B)

    # march in time
    for n in range(Nt - 1):
        # operate on indices 1..Nx-1 (i.e., skip index 0 which will be set = index - periodic)
        u[1:, n + 1] = C @ u[1:, n]
        # enforce periodicity: first grid point equals last
        u[0, n + 1] = u[-1, n + 1]
    return x, t, u


if __name__ == "__main__":
    # quick smoke test (analytical solution for V(x)=sin(2pi x) with advection speed 1 and diffusion D)
    xmin, xmax = 0.0, 1.0
    tmin, tmax = 0.0, 0.1
    Nx, Nt = 201, 201
    D = 0.1
    V = lambda x: np.sin(2 * np.pi * x)
    x, t, u = solve_ADVD(xmin, xmax, tmin, tmax, None, None, V, Nx, Nt, D=D)
    # analytical reference for u(x,t) = exp(-4 pi^2 D t) * sin(2 pi (x - t))
    xs = x[:, None]
    ts = t[None, :]
    u_true = np.exp(-4 * np.pi ** 2 * D * ts) * np.sin(2 * np.pi * (xs - ts))
    # compute error at final time
    err = np.max(np.abs(u - u_true))
    print("Max abs error (advd) approx:", err)
