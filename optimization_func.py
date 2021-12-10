import numpy as np
import matplotlib.pyplot as plt

from helper import int_cost_lotka_volterra


def simulated_annealing(s0, t, P,
    cost=int_cost_lotka_volterra, cooling_schedule="linear",
    T_start=100, T_steps=500, sigma=0.1
):
    if cooling_schedule == "linear":
        T_sched = np.linspace(T_start, 0.1, T_steps)
    state = s0
    state_cost = cost(state, P, t)

    for T in T_sched:
        state_new = np.random.normal(state, scale=sigma)
        U = np.random.rand()

        new_state_cost = cost(state_new, P, t)
        threshold = np.min([
            np.exp(-(new_state_cost - state_cost)/T),
            1
        ])

        if U <= threshold:
            state = state_new
            state_cost = new_state_cost

    return state
