from optimization_func import simulated_annealing
from helper import mse_trunc, int_cost_lotka_volterra
import queue

def vary_T_worker(q, d, T_start, t_data, P, cooling_schedule):
    while True:
        try:
            t, i, rv = q.get_nowait()
        except queue.Empty:
            break

        _, _, cost = simulated_annealing(rv, t_data, P, T_start=T_start, T_steps=t, cooling_schedule=cooling_schedule)
        d[i].append(cost)


def vary_sigma_worker(q, d, T_start, T_steps, t_data, P, cooling_schedule):
    while True:
        try:
            s, i, rv = q.get_nowait()
        except queue.Empty:
            break

        _, _, cost = simulated_annealing(rv, t_data, P, T_start=T_start, T_steps=T_steps, cooling_schedule=cooling_schedule,eta=s)
        d[i].append(cost)

def vary_rv_worker(q, d_cost, d_state, T_start, T_steps, t_data, P):
    while True:
        try:
            i, rv = q.get_nowait()
        except queue.Empty:
            break

        state, _, cost = simulated_annealing(rv, t_data, P, T_start=T_start,
                T_steps=T_steps, cooling_schedule="quadratic", eta=0.2)
        d_cost[i].append(cost)
        d_state[i].append(state)

def vary_truncation_worker(q, d, T_start, T_steps, true_P, t_data, rv, datasets):
    while True:
        try:
            i, _ = q.get_nowait()
        except queue.Empty:
            break

        P = datasets[i]
        x, _, cost = simulated_annealing(rv, t_data, P, y0=true_P[0],
                T_start=T_start, T_steps=T_steps, cooling_schedule="quadratic",
                eta=0.2, cost=mse_trunc)
        true_cost = int_cost_lotka_volterra(x, true_P, t_data, y0=true_P[0])
        d[i].append(true_cost)
