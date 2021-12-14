import numpy as np
import matplotlib.pyplot as plt

from helper import int_cost_lotka_volterra, mse, mae, r2


def simulated_annealing(s0, t, P,
        cost=mse, cooling_schedule="geometrical",
        T_start=100, T_steps=500, eta=0.1, alpha=0.95, write_costs = False
        ):
    if cooling_schedule == "geometrical":
        T_sched =  [T_start*alpha**k for k in range(T_steps)]
    if cooling_schedule == "linear":
        T_sched = np.linspace(T_start, 1/T_start, T_steps)
    if cooling_schedule == "quadratic":
        T_sched = [T_start/(1 + alpha*k**2) for k in range(T_steps)]

    state = s0
    state_cost = int_cost_lotka_volterra(state, P, t, cost)
    costs = []

    n = 0
    n_accepted = 0

    for T in T_sched:
        costs.append(state_cost)

        state_new = np.zeros(len(state))
        for i in range(len(state)):
            while True:
                new_param = np.random.normal(state[i], scale=eta*state[i])
                if new_param >= 0:
                    state_new[i] = new_param
                    break

        new_state_cost = int_cost_lotka_volterra(state_new, P, t, cost)
        if new_state_cost < state_cost:
            state = state_new
            state_cost = new_state_cost
            continue

        n += 1
        threshold = np.min([
            np.exp(-(new_state_cost - state_cost)/T),
            1
            ])

        U = np.random.rand()

        if U <= threshold:
            state = state_new
            state_cost = new_state_cost
            n_accepted += 1

    costs.append(state_cost)

    if write_costs == True:
        return state, n_accepted/n, state_cost, costs
    else:
        return state, n_accepted/n, state_cost
