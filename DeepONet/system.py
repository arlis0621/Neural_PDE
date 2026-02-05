# system.py
import numpy as np
from scipy import interpolate
from scipy.integrate import quad
from utils import timing
from cvc_solver import solve_CVC
from advd_solver import solve_ADVD
from avr_solver import solve_AVR
import numpy as np

# def gen_pde_operator_data(
#     space,               # function-space object (e.g., GRF, FinitePowerSeries)
#     problem_type,        # 'CVC' | 'ADVD' | 'AVR'
#     m,                   # number of spatial sensors for input V
#     num_samples,         # number of operator examples to generate
#     Nx=128,              # spatial grid resolution for solver
#     Nt=128,              # temporal grid resolution for solver
#     T=1.0,               # final time
#     query_per_sample=1,  # how many (x,t) queries per sample (1 => one query per sample)
#     method='analytic',   # solver method option string forwarded to solver
#     periodic=True,
# ):
#     """
#     Generate operator dataset for PDE problems.

#     Returns:
#       (X_sensor_vals, X_queries), y_targets
#       - X_sensor_vals: (num_samples, m) values of V at m sensor x-locations on [0,T_space]
#       - X_queries: (num_samples * query_per_sample, 2) array of (x_query, t_query)
#       - y_targets: (num_samples * query_per_sample, 1) corresponding u(x_query,t_query)
#     """
#     # define spatial sensors for input V (choose uniform sensors over [0,1])
#     xs_sensors = np.linspace(0.0, 1.0, m)

#     # pre-allocate lists
#     all_sensor_vals = []
#     all_queries = []
#     all_targets = []

#     # choose solver & solver-args
#     for i in range(num_samples):
#         # 1) sample a random input function V
#         feat = space.random(1)[0]        # e.g., GRF.random(1) -> shape (1, N) -> take [0]
#         # evaluate V on a high-res spatial grid for the solver
#         x_grid = np.linspace(0.0, 1.0, Nx)
#         V_on_grid = space.eval_u_one(feat, x_grid)  # (Nx,) vector

#         # 2) run PDE solver to get u(x,t) on grid
#         if problem_type == 'CVC':
#             xg, tg, u_grid = solve_CVC(0.0, 1.0, 0.0, T, f=None, g=None, V=lambda z: np.interp(z, x_grid, V_on_grid), Nx=Nx, Nt=Nt, method=method, periodic=periodic)
#         elif problem_type == 'ADVD':
#             xg, tg, u_grid = solve_ADVD(0.0, 1.0, 0.0, T, f=None, g=None, V=lambda z: np.interp(z, x_grid, V_on_grid), Nx=Nx, Nt=Nt)
#         elif problem_type == 'AVR':
#             xg, tg, u_grid = solve_AVR(0.0, 1.0, 0.0, T, V=lambda z: np.interp(z, x_grid, V_on_grid), Nx=Nx, Nt=Nt, method=method)
#         else:
#             raise ValueError("Unknown problem_type: " + str(problem_type))

#         # 3) compute sensor values for input V at the m sensor locations (these form the branch input)
#         sensor_vals = space.eval_u_one(feat, xs_sensors)  # shape (m,)
#         # append (converted to 1d)
#         for q in range(query_per_sample):
#             # choose a random query time-space point (xq, tq) or choose structured points as desired
#             xq = np.random.choice(xg)   # pick from the solver spatial grid
#             tq = np.random.choice(tg)   # pick from the solver time grid
#             # locate nearest grid indices
#             ix = int(np.searchsorted(xg, xq, side='left'))
#             it = int(np.searchsorted(tg, tq, side='left'))
#             # clip indices into valid range
#             ix = min(max(ix, 0), len(xg)-1)
#             it = min(max(it, 0), len(tg)-1)
#             yq = float(u_grid[ix, it])   # u(xq, tq)
#             all_sensor_vals.append(sensor_vals.copy())
#             all_queries.append([xq, tq])
#             all_targets.append([yq])

#     X_sensor_vals = np.vstack(all_sensor_vals)               # (num_samples*query_per_sample, m)
#     X_queries = np.array(all_queries).reshape(-1, 2)         # (num_samples*query_per_sample, 2)
#     y_targets = np.array(all_targets).reshape(-1, 1)         # (num_samples*query_per_sample, 1)
#     return (X_sensor_vals, X_queries), y_targets

#Implements the ODESystem used by the original repo.For the Antiderivative case, g(s, u, x) = u, s0 = [0].
# We compute s(tf) = integral_0^{tf} u(t) dt.
    
class ODESystem:
    

    def __init__(self, g, s0, T):
        self.g = g
        self.s0 = s0
        self.T = T
#Generate operator data for 'num' random input functions.Returns: (X_inputs, X_t), y
# - X_inputs: array (num, m) sensor values for each function
# - X_t: array (num, 1) random tf for each function (where s(tf) is evaluated)
# - y: array (num, 1) s(tf)
    @timing
    def gen_operator_data(self, space, m, num):
        
        print("Generating operator data...", flush=True)
        features = space.random(num)  # e.g., GRF.random -> (num, N)
        sensors = np.linspace(0, self.T, num=m)
        sensor_values = space.eval_u(features, sensors[:, None])  # (num, m)

        # choose random evaluation times tf in [0, T]
        x = self.T * np.random.rand(num)
        # for each feature, evaluate s(tf) by integrating u from 0 to tf
        y = self.eval_s_space(space, features, x)
        # shapes:
        # sensor_values: (num, m)
        # x: (num,)
        # y: (num, 1)
        return [sensor_values, x.reshape(-1, 1)], y.reshape(-1, 1)
    
    
    
#Compute s(tf) for list of features and corresponding tf values.
    def eval_s_space(self, space, features, x_array):
        
        res = []
        for feat, tf in zip(features, x_array):
            # convert feat -> continuous function u(t) via space.eval_u_one
            def ufun(t):
                return float(space.eval_u_one(feat, t))
            res.append(self.eval_s(ufun, tf))
        return np.array(res)
    
    
# Compute s(tf) by numerically integrating u from 0 to tf:s(tf) = s(0) + integral_0^{tf} g(s(t), u(t), t) dt
#For antiderivative case g(s, u, t) = u, s0=0 => s(tf)=integral_0^{tf} u(t) dt
#We implement a scalar integral of u(t) dt.
    
    def eval_s(self, u, tf):
        # simple quadrature using scipy.integrate.quad
        if tf <= 0:
            return np.array([self.s0[0]])
        val, _ = quad(lambda t: float(u(t)), 0.0, float(tf), epsabs=1e-8, epsrel=1e-8, limit=200)
        return np.array([self.s0[0] + val])
