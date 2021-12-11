import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp


def get_predator_prey_data():
    df = pd.read_csv("data/predator-prey-data.csv")
    t = np.array(df["t"])
    y = np.array(df[["x", "y"]])

    return t, y


def mse(y_actual, y_pred):
    assert y_actual.shape == y_pred.shape, "Actual and prediction shapes mismatch"

    return (np.linalg.norm(y_pred - y_actual, axis=1)**2).mean()


def lotka_volterra(t, ys, alpha, beta, delta, gamma):
    x, y = ys
    dx_dt = alpha*x - beta*x*y
    dy_dt = delta*x*y - gamma*y

    return dx_dt, dy_dt


def int_cost_lotka_volterra(params, y_actual, ts, cost=mse):
    sol = solve_ivp(lotka_volterra, (ts[0], ts[-1]), y_actual[0], args=params,
                    dense_output=True, max_step = 1e3)
    z = sol.sol(ts)

    return cost(y_actual, z.T)
