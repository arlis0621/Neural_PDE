# cvc_solver.py
"""
Conservative/Variable-coefficient Convection (CVC) solver utilities.

Functions:
- solve_CVC: solve u_t + a(x)*u_x = 0 on [xmin,xmax] x [tmin,tmax]
    Options:
      - method='analytic' : returns analytic solution for a(x)=1 and periodic V.
      - method='wendroff' : Lax–Wendroff / Wendroff-style solver for constant or variable a(x).
    Inputs:
      xmin, xmax, tmin, tmax : domain
      f : initial condition function f(x) if provided (used for inflow cases)
      g : boundary inflow u(0,t) if provided (used for inflow cases)
      V : if provided and method=='analytic', V(x) used as initial periodic profile (or velocity profile if variable cases)
      Nx, Nt : grid sizes in x and t
      a_func : optional callable a(x) specifying the velocity field (if None, a(x)=1)
      periodic : bool; if True uses periodic BCs; otherwise uses inflow at x= xmin given by g(t)
      verbose : bool; print debug info
Returns:
    x (Nx,), t (Nt,), u (Nx, Nt) array
"""
import numpy as np


def solve_CVC(
    xmin,
    xmax,
    tmin,
    tmax,
    f=None,
    g=None,
    V=None,
    Nx=100,
    Nt=100,
    a_func=None,
    method="analytic",
    periodic=True,
    verbose=False,
):
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]

    if a_func is None:
        def a_func_local(xv): return np.ones_like(xv)
        a_func = a_func_local

    # Analytic solution for a(x)=1 and periodic initial data V:
    if method == "analytic":
        if V is None:
            raise ValueError("Analytic method requires V (periodic initial profile).")
        # analytic solution u(x,t) = V(x - t) under a(x)=1, periodic domain assumed on length L = xmax-xmin
        L = xmax - xmin
        # map domain to [0, L]
        xs = x[:, None]
        ts = t[None, :]
        # compute shifted coords modulo domain length
        coords = (xs - ts - xmin) % L + xmin
        u = V(coords)
        if verbose:
            print("solve_CVC: analytic solution using shift (a=1, periodic V).")
        return x, t, u

    # Numerical Wendroff-like / Lax-Wendroff implementation (second-order)
    # We implement a conservative Lax-Wendroff for u_t + a(x)*u_x = 0.
    # For variable a(x) we use local face speeds a_{i+1/2} = 0.5*(a_i + a_{i+1})
    if method == "wendroff":
        if f is None and V is None:
            raise ValueError("wendroff method requires an initial profile: f (or V) provided.")
        # initial condition
        if f is None:
            u0 = V(x)
        else:
            u0 = f(x)
        u = np.zeros((Nx, Nt))
        u[:, 0] = u0.copy()

        # precompute a(x) at cell centers and faces
        a_centers = a_func(x)
        # face speeds at i+1/2 between x[i] and x[i+1]
        a_faces = 0.5 * (a_centers[:-1] + a_centers[1:])

        # CFL stability: dt * max|a| / h <= 1
        max_a = np.max(np.abs(a_centers))
        if max_a * dt / h > 1.0:
            if verbose:
                print("Warning: CFL condition violated (|a|max * dt / dx = {:.3f} > 1).".format(max_a * dt / h))

        # periodic or inflow
        if periodic:
            # periodic indexing
            for n in range(Nt - 1):
                # compute fluxes using Lax-Wendroff:
                # u_i^{n+1} = u_i^n - (dt/(2h))*(a_{i+1}*(u_{i+1}-u_{i}) + a_{i}*(u_{i}-u_{i-1}))
                #          + (dt**2)/(2*h**2)*(a_{i+1}**2*(u_{i+1}-2u_i+u_{i-1}) - a_{i}**2*(u_{i}-2u_{i-1}+u_{i-2}))
                un = u[:, n]
                # roll indices for periodic neighbors
                up = np.roll(un, -1)
                um = np.roll(un, 1)
                a_i = a_centers
                a_ip = np.roll(a_centers, -1)
                term1 = (dt / (2 * h)) * (a_ip * (up - un) + a_i * (un - um))
                # second-order correction approximate using central second differences
                # Use scalar squared a centered
                a2 = a_centers ** 2
                a2_ip = np.roll(a2, -1)
                a2_im = np.roll(a2, 1)
                term2 = (dt ** 2) / (2 * h ** 2) * (a2_ip * (up - 2 * un + um) - a2_im * (un - 2 * um + np.roll(um, 1)))
                u[:, n + 1] = un - term1 + term2
            # ensure periodicity at boundaries (already enforced by roll)
            return x, t, u

        else:
            # inflow boundary at x[0] given by g(t); left boundary x= xmin.
            # Use one-sided differences near left boundary.
            for n in range(Nt - 1):
                un = u[:, n].copy()
                up = np.zeros_like(un)
                um = np.zeros_like(un)
                up[:-1] = un[1:]
                up[-1] = un[-1]  # for last cell use zero-gradient
                um[1:] = un[:-1]
                um[0] = g(n * dt) if g is not None else un[0]
                a_i = a_centers
                a_ip = np.concatenate((a_centers[1:], a_centers[-1:]))
                term1 = (dt / (2 * h)) * (a_ip * (up - un) + a_i * (un - um))
                a2 = a_centers ** 2
                a2_ip = np.concatenate((a2[1:], a2[-1:]))
                a2_im = np.concatenate((a2[:1], a2[:-1]))
                term2 = (dt ** 2) / (2 * h ** 2) * (a2_ip * (up - 2 * un + um) - a2_im * (un - 2 * um + np.concatenate((um[:1], um[:-1]))))
                u[:, n + 1] = un - term1 + term2
            return x, t, u

    raise ValueError("Unknown method '{}'. Use 'analytic' or 'wendroff'.".format(method))


if __name__ == "__main__":
    # quick smoke test for analytic case
    xmin, xmax = 0.0, 1.0
    tmin, tmax = 0.0, 1.0
    Nx, Nt = 101, 101
    V = lambda z: np.sin(2 * np.pi * z)
    x, t, u = solve_CVC(xmin, xmax, tmin, tmax, f=None, g=None, V=V, Nx=Nx, Nt=Nt, method="analytic", periodic=True)
    # check that u(x,t) ~= V(x - t)
    xs = x[:, None]
    ts = t[None, :]
    u_true = V((xs - ts) % 1.0)
    print("Max abs error analytic:", np.max(np.abs(u - u_true)))