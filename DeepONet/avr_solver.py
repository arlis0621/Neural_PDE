# avr_solver.py
"""
AVR solver (average-velocity advection solver).

This module provides a lightweight solver for advection using the domain-average velocity:
    u_t + a_avg * u_x = 0

It is useful as a simple baseline / approximation when original code refers to 'avr' solver.
Functions:
- solve_AVR(xmin, xmax, tmin, tmax, V, Nx, Nt, periodic=True, method='analytic'/'upwind')
    V : initial profile callable V(x)
    method:
      - 'analytic' : if a_avg is constant and periodic domain, solution is V(x - a_avg * t)
      - 'upwind'   : explicit upwind finite-difference (first-order), stable under CFL.
Returns x (Nx,), t (Nt,), u (Nx, Nt)
"""
import numpy as np


def solve_AVR(xmin, xmax, tmin, tmax, V, Nx=100, Nt=100, periodic=True, method="analytic", a_func=None, verbose=False):
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]

    if a_func is None:
        # default unit velocity
        a_func = lambda xx: np.ones_like(xx)

    a_centers = a_func(x)
    a_avg = float(np.mean(a_centers))

    if method == "analytic":
        # analytic shift with average velocity
        L = xmax - xmin
        xs = x[:, None]
        ts = t[None, :]
        coords = (xs - a_avg * ts - xmin) % L + xmin
        u = V(coords)
        if verbose:
            print(f"solve_AVR: analytic shift with a_avg={a_avg:.4f}")
        return x, t, u

    elif method == "upwind":
        u = np.zeros((Nx, Nt))
        u[:, 0] = V(x)
        maxvel = abs(a_avg)
        if maxvel * dt / h > 1.0 and verbose:
            print("Warning: CFL violated (a_avg*dt/dx > 1).")

        for n in range(Nt - 1):
            un = u[:, n].copy()
            if a_avg >= 0:
                # upwind: u_i^{n+1} = u_i^n - (a_avg * dt / h) * (u_i^n - u_{i-1}^n)
                u[1:, n + 1] = un[1:] - (a_avg * dt / h) * (un[1:] - un[:-1])
                # periodic left case
                u[0, n + 1] = un[0] - (a_avg * dt / h) * (un[0] - un[-1])
            else:
                # negative velocity: upwind from right
                u[:-1, n + 1] = un[:-1] - (a_avg * dt / h) * (un[1:] - un[:-1])
                u[-1, n + 1] = un[-1] - (a_avg * dt / h) * (un[0] - un[-1])
        return x, t, u

    else:
        raise ValueError("Unknown method for solve_AVR: " + str(method))


if __name__ == "__main__":
    # smoke test
    xmin, xmax = 0.0, 1.0
    tmin, tmax = 0.0, 1.0
    Nx, Nt = 101, 101
    V = lambda z: np.sin(2 * np.pi * z)
    x, t, u = solve_AVR(xmin, xmax, tmin, tmax, V, Nx=Nx, Nt=Nt, periodic=True, method="analytic")
    # compare to V(x - t) since a_avg should be 1 if a_func default returns 1
    xs = x[:, None]
    ts = t[None, :]
    u_true = V((xs - ts) % 1.0)
    print("Max abs error AVR analytic:", np.max(np.abs(u - u_true)))
