import numpy as np
import matplotlib.pyplot as plt

from helper import int_cost_lotka_volterra


def simulated_annealing(s0, t, P,
    cost=int_cost_lotka_volterra, cooling_schedule="geometrical",
    T_start=100, T_steps=500, sigma=0.1, alpha=1
):
    if cooling_schedule == "geometrical":
        T_sched =  [T_start*alpha**k for k in range(T_steps)]
    if cooling_schedule == "linear":
        T_sched = np.linspace(T_start, 1e-5, T_steps)
    if cooling_schedule == "Quadratic":
        T_sched = [T_start/(1 + alpha*k**2) for k in range(T_steps)]

    state = s0
    state_cost = cost(state, P, t)

    n_accepted = 0

    for T in T_sched:
        state_new = np.zeros(len(state))
        for i in range(len(state)):
            while True:
                new_param = np.random.normal(state[i], scale=sigma)
                if new_param >= 0:
                    state_new[i] = new_param
                    break

        new_state_cost = cost(state_new, P, t)
        if new_state_cost < state_cost:
            state = state_new
            state_cost = new_state_cost
            n_accepted += 1
            continue

        threshold = np.min([
            np.exp(-(new_state_cost - state_cost)/T),
            1
        ])

        U = np.random.rand()

        if U <= threshold:
            state = state_new
            state_cost = new_state_cost
            n_accepted += 1

    return state, n_accepted/len(T_sched)
