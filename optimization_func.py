from sklearn.metrics import mean_squared_error as MSE
import numpy as np
import matplotlib.pyplot as plt



def Simulated_Annealing(s0, Cooling_schedule = "linear", Opt_function = <SE, T_start = 100, sigma = 0.1, data = (t_range, True_data)):
    if Cooling_schedule == "linear":
        T_sched = np.linspace(100, 0.1, 5000)
    state = s0
    pred = f(state, data[0])

    for T in T_sched:
        state_new = np.random.normal(state, scale = sigma)
        U = np.random.rand()

        pred_new = f(state_new, data[0])

        threshold = np.min(np.exp(-(Opt_function(data[1], pred_new) - Opt_function(data[1], pred))/T))
        if U <= threshold:
            state = state_new
            pred = pred_new 
        else: 
            state = state
    
    return state