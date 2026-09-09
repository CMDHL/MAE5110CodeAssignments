import numpy as np


def dynamics(t, state, params):
    g = params["gravity"]
    m = params["mass"]
    k = params["k"]
    h = state[0]
    v = state[1]

    a = -g
    if h < 0:
        a = -g-k*h/m
    return np.array([v, a])


def generate_params():
    params = {
        "gravity": 9.81,  # gravity m/s^2)
        "mass": 1,  # point mass (kg)
        "k":1000, # spring constant (kg/s^2)
    }
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    g = params["gravity"]
    m = params["mass"]
    k = params["k"]
    h = state[0]
    v = state[1]
    kinetic_energy = 0.5 * m * v**2
    potential_energy = m*g*h + np.where(h<0, 0.5*k*h**2, 0)
    return kinetic_energy, potential_energy
