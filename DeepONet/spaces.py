# spaces.py
import numpy as np
from scipy import interpolate
from sklearn import gaussian_process as gp




class FinitePowerSeries:
    def __init__(self, N=20, M=1):
        self.N = N
        self.M = M

    def random(self, n):
        # coefficients in [-M, M]
        return 2 * self.M * np.random.rand(n, self.N) - self.M

    def eval_u_one(self, a, x):
        # x can be array
        powers = np.vstack([x.ravel() ** i for i in range(self.N)])  # N x len(x)
        return np.dot(a, powers).ravel()

    def eval_u(self, as_, sensors):
        # as_: (n, N), sensors: (m,1) or (m,)
        sensors = np.ravel(sensors)
        mat = np.vstack([sensors ** i for i in range(self.N)])  # N x m
        return np.dot(as_, mat).reshape((as_.shape[0], len(sensors)))


#Difference betwween eval_u_one and eval_u?
#Ans. eval_u_one evaluates a single function defined by coefficients 'a' at points 'x', while 
#eval_u evaluates multiple functions defined by coefficients in 'as_' at points 'sensors'


class FiniteChebyshev:
    def __init__(self, N=20, M=1):
        self.N = N
        self.M = M

    def random(self, n):
        return 2 * self.M * np.random.rand(n, self.N) - self.M

    def eval_u_one(self, a, x):
        xs = 2 * np.ravel(x) - 1.0
        return np.polynomial.chebyshev.chebval(xs, a).ravel()

    def eval_u(self, as_, sensors):
        sensors = np.ravel(sensors)
        # using broadcasting: outputs shape (n, len(sensors))
        return np.array([np.polynomial.chebyshev.chebval(2 * sensors - 1, a) for a in as_])

#Gaussian random field sampler on [0, T] discretized to N grid points.
# Implementation similar to original repo: precompute kernel matrix and L for sampling,
# then interpolate to requested sensor locations.
class GRF:
    

    def __init__(self, T=1.0, kernel="RBF", length_scale=0.2, N=200, interp="cubic"):
        self.T = T
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)
        self.length_scale = length_scale
        # choose kernel from sklearn
        if kernel == "RBF":
            Kfun = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            Kfun = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        else:
            Kfun = gp.kernels.RBF(length_scale=length_scale)
        self.K = Kfun(self.x[:, None])
        # add small jitter
        jitter = 1e-12 * np.eye(self.N)
        try:
            self.L = np.linalg.cholesky(self.K + jitter)
        except np.linalg.LinAlgError:
            # fallback to eigh
            w, v = np.linalg.eigh(self.K + jitter)
            w = np.clip(w, 1e-14, None)
            self.L = (v * np.sqrt(w)).dot(np.eye(self.N))
#Return (n, N) array of samples on the internal grid.
    def random(self, n):
        
        z = np.random.randn(self.N, n)
        samples = (self.L @ z).T  # shape (n, N)
        return samples
#Evaluate a single sampled function y (length N) at points x (scalar or array).
    def eval_u_one(self, y, x):
        
        x = np.ravel(x)
        if self.interp == "linear":
            return np.interp(x, self.x, y)
        f = interpolate.interp1d(self.x, y, kind=self.interp, fill_value="extrapolate", assume_sorted=True)
        return f(x)
#ys: (n, N), sensors: (m,1) or (m,) returns (n, m) array of evaluations at sensors
        
    def eval_u(self, ys, sensors):
        
        sensors = np.ravel(sensors)
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, self.x, y) for y in ys])
        res = []
        for y in ys:
            f = interpolate.interp1d(self.x, y, kind=self.interp, fill_value="extrapolate", assume_sorted=True)
            res.append(f(sensors))
        return np.vstack(res)
