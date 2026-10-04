import numpy as np

from integrators import rk4
from models import pendulum


def test_energy_conservation():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0

    state = np.array([0.5, 0.0])

    timestep = 0.01
    energies = []

    for step in range(100):
        kinetic, potential = pendulum.calculate_energy(state, params)
        energies.append(kinetic + potential)

        state = rk4(pendulum.dynamics,step * timestep,state,timestep,params,)

    energy_change = np.array(energies) - energies[0]

    assert np.all(np.isclose(energy_change, 0.0, atol=1e-6))


def test_torque_adds_energy():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 1.0

    state = np.array([0.5, 1.0])

    kinetic_initial, potential_initial = pendulum.calculate_energy(state, params)
    initial_energy = kinetic_initial + potential_initial

    timestep = 0.01

    for step in range(100):
        state = rk4(pendulum.dynamics,step * timestep,state,timestep,params,)

    kinetic_final, potential_final = pendulum.calculate_energy(state, params)
    final_energy = kinetic_final + potential_final

    assert final_energy > initial_energy


def test_damping_removes_energy():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.1
    params["torque"] = 0.0

    state = np.array([0.5, 1.0])

    kinetic_initial, potential_initial = pendulum.calculate_energy(state, params)
    initial_energy = kinetic_initial + potential_initial

    timestep = 0.01

    for step in range(100):state = rk4(pendulum.dynamics,step * timestep,state,timestep,params,)

    kinetic_final, potential_final = pendulum.calculate_energy(state, params)
    final_energy = kinetic_final + potential_final

    assert final_energy < initial_energy