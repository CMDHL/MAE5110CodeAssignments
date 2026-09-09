import numpy as np
import matplotlib.pyplot as plt
import timeit

from models import rimless_wheel as model
# from integrators import explicit_euler as integrator
from integrators import rk4 as integrator

def sim(initial_state, timestep, sim_time, params, model):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for i, t in enumerate(time_traj[:-1]):
        next_state = integrator.step(t, state_traj[:, i], timestep, params, model)
        if model.need_reset(next_state, params):
        	next_state = model.reset_state(next_state, params)
        state_traj[:, i + 1] = next_state
    return time_traj, state_traj

params = model.generate_params()
initial_state = np.array([0, 1.5])
sim_time = 10.0

timestep = 1e-3
time_traj, state_traj = sim(initial_state, timestep, sim_time, params, model)
kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

plt.figure()
angle,angular_velocity = state_traj
plt.plot(time_traj,angle,label="angle (rad)")
plt.plot(time_traj,angular_velocity,label="angular velocity (rad/s)")
plt.xlabel("Time (s)")
plt.title("state trajectories")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(angle, angular_velocity)
plt.xlabel("Angle (rad)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Phase portrait")
plt.tight_layout()

plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.legend()
plt.tight_layout()
plt.show()