import numpy as np
import matplotlib.pyplot as plt
import timeit 

from models import pendulum as model
from integrators import explicit_euler as integrator
# from integrators import rk4 as integrator

# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}

initial_state = np.array([np.pi / 4, 0.0])
sim_time = 5.0

for timestep in [1e-5,1e-4,1e-3,1e-2,1e-1,1]:
    time_traj, state_traj = integrator.sim(initial_state, timestep, sim_time, params, model)
    potential_energy, kinetic_energy = model.calculate_energy(state_traj, params)
    print(timestep, '\t', np.ptp(potential_energy + kinetic_energy))

    plt.figure()
    plt.plot(time_traj, potential_energy, label="Potential energy")
    plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
    plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.title("Pendulum energy")
    plt.legend()
    plt.tight_layout()
    plt.show()

# TODO: make a phase portrait plot
