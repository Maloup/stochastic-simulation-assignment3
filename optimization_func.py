import numpy as np
import matplotlib.pyplot as plt

from helper import int_cost_lotka_volterra


def simulated_annealing(s0, t, P,
    cost=int_cost_lotka_volterra, cooling_schedule="linear",
    T_start=100, T_steps=500, sigma=0.1
):
    if cooling_schedule == "linear":
        #T_sched = np.linspace(T_start, 0.1, T_steps)
        #T_sched =  [(T_start/(1 + 20*np.log(1 + k))) for k in range(T_steps)]
        T_sched =  [(T_start/(1 + k)) for k in range(T_steps)]
        #T_sched = 1/np.linspace(0.001, T_start, T_steps)**1.1
    state = s0
    state_cost = cost(state, P, t)

    for T in T_sched:
        state_new = np.zeros(len(state))
        for i in range(len(state)):
            while True:
                new_param = np.random.normal(state[i], scale=sigma)
                if new_param >= 0:
                    state_new[i] = new_param
                    break

        #sstate_new = np.random.normal(state, scale=sigma)
        U = np.random.rand()

        new_state_cost = cost(state_new, P, t)
        threshold = np.min([
            np.exp(-(new_state_cost - state_cost)/T),
            1
        ])

        print(threshold)

        if U <= threshold:
            state = state_new
            state_cost = new_state_cost

    return state
