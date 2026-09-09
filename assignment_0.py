import numpy as np
import matplotlib.pyplot as plt
import timeit

# from models import pendulum as model
from models import bouncing_ball as model
# from integrators import explicit_euler as integrator
from integrators import rk4 as integrator

# Basic simulation of the pendulum


def sim(initial_state, timestep, sim_time, params, model):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for i, t in enumerate(time_traj[:-1]):
        state_traj[:, i + 1] = integrator.step(t, state_traj[:, i], timestep, params, model)
    return time_traj, state_traj


params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}

initial_state = np.array([np.pi / 4, 0.0])
sim_time = 5.0
    
# threshold = 0.1
# max_step = 1e-5

# for timestep in [1e-5,1e-4,1e-3,1e-2,1e-1,1]:
#     time_traj, state_traj = sim(initial_state, timestep, sim_time, params, model)
#     kinetic_energy, potential_energy= model.calculate_energy(state_traj, params)
#     diff = np.ptp(potential_energy + kinetic_energy)
#     print(timestep, '\t', diff)
#     if diff<threshold:
#         max_step = timestep
#     else:
#         break

# print("max step:", max_step)

# t1 = timeit.default_timer()
# time_traj, state_traj = sim(initial_state, 1e-5, sim_time, params, model)
# t2 = timeit.default_timer()
# print("initial:\t", t2-t1)
# kinetic_energy, potential_energy  = model.calculate_energy(state_traj, params)

# t1 = timeit.default_timer()
# sim(initial_state, max_step, sim_time, params, model)
# t2 = timeit.default_timer()
# print("largest:\t", t2-t1)


# DONE: make a phase portrait plot
# plt.figure()
# x,y = state_traj
# plt.plot(x, y)
# plt.title("phase portrait plot")
# plt.legend()
# plt.tight_layout()
# plt.show()

params = model.generate_params()
initial_state = np.array([10, 0.0])
timestep = 1e-2
time_traj, state_traj = sim(initial_state, timestep, sim_time, params, model)
kinetic_energy,potential_energy = model.calculate_energy(state_traj, params)

plt.figure()
h,v = state_traj
plt.plot(time_traj,h,label="h")
plt.plot(time_traj,v,label="v")
plt.xlabel("Time (s)")
plt.legend()
plt.tight_layout()
plt.show()

# plt.figure()
# plt.plot(time_traj, potential_energy, label="Potential energy")
# plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
# plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
# plt.xlabel("Time (s)")
# plt.ylabel("Energy (J)")
# plt.title("Pendulum energy")
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure()
# angle, _ = state_traj
# plt.plot(time_traj, angle,label="angle")
# plt.xlabel("Time (s)")
# plt.legend()
# plt.tight_layout()
# plt.show()
