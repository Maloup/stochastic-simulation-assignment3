import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from sklearn.metrics import r2_score

def get_predator_prey_data():
    df = pd.read_csv("data/predator-prey-data.csv")
    t = np.array(df["t"])
    y = np.array(df[["x", "y"]])

    return t, y


def mse(y_actual, y_pred):
    assert y_actual.shape == y_pred.shape, "Actual and prediction shapes mismatch"

    return (np.linalg.norm(y_pred - y_actual, axis=1)**2).mean()


def mse_trunc(y_actual, y_pred):
    assert y_actual.shape == y_pred.shape, "Actual and prediction shapes mismatch"

    M = y_actual.shape[0]
    n = 0
    cost = 0

    for i in range(M):
        true_x, true_y = y_actual[i]
        pred_x, pred_y = y_pred[i]

        if true_x != -1:
            cost += (true_x - pred_x)**2
            n += 1

        if true_y != -1:
            cost += (true_y - pred_y)**2
            n += 1

    if n == 0:
        return 0

    return cost/n


def mae(y_actual, y_pred):
    assert y_actual.shape == y_pred.shape, "Actual and prediction shapes mismatch"

    return np.abs(y_pred - y_actual).mean()

def r2(y_actual, y_pred):
    assert y_actual.shape == y_pred.shape, "Actual and prediction shapes mismatch"

    return r2_score(y_actual, y_pred)


def lotka_volterra(t, ys, alpha, beta, delta, gamma):
    x, y = ys
    dx_dt = alpha*x - beta*x*y
    dy_dt = delta*x*y - gamma*y

    return dx_dt, dy_dt


def int_cost_lotka_volterra(params, y_actual, ts, cost=mse, y0=None):
    if y0 is None:
        y0 = y_actual[0]
    sol = odeint(lotka_volterra, y0, ts, args=tuple(params), tfirst=True)

    return cost(y_actual, sol)

def remove_data_points_rand(data, n, col=0):
    col_data = data[:,col]
    idxs = np.argwhere(col_data != -1).flatten()
    removed_idxs = np.random.choice(idxs, size=n, replace=False)
    col_data[removed_idxs] = -1

    return data

def remove_data_points_det(data, n, col=0):
    col_data = data[:,col]
    i = 0
    while col_data[i] == -1:
        i += 1

    for j in range(n):
        col_data[i+j] = -1

    return data
