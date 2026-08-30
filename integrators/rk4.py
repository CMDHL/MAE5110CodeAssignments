import numpy as np

def sim(initial_state, timestep, sim_time, params, model):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    
    h = timestep

    for step, t in enumerate(time_traj[:-1]):
        y = state_traj[:, step]
        k1 = model.dynamics(t, y, params)
        k2 = model.dynamics(t+h/2, y+k1*h/2, params)
        k3 = model.dynamics(t+h/2, y+k2*h/2, params)
        k4 = model.dynamics(t+h, y+h*k3, params)
        state_traj[:, step + 1] = y+h/6*(k1+2*k2+2*k3+k4)
    return time_traj, state_traj
    