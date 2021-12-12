from optimization_func import simulated_annealing
import queue

def vary_T_worker(q, d, T_start, t_data, P, cooling_schedule):
    while True:
        try:
            t, i, rv = q.get_nowait()
        except queue.Empty:
            break

        _, _, cost = simulated_annealing(rv, t_data, P, T_start=T_start, T_steps=t, cooling_schedule=cooling_schedule)
        d[i].append(cost)
