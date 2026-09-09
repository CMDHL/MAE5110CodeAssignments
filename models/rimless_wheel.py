import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
        - damping_coeff * angular_velocity  # <-- DAMPING TERM
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def generate_params():
    params = {
        "N": 8, # number of spokes
        "gamma": 0.1, # downhill inclination in rad
        "gravity": 9.81,  # gravity m/s^2)
        "length": 1,  # spoke length (m)
        "mass": 1,  # point mass at hub (kg)
        "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
    }
    params['alpha'] = np.pi / params['N']
    return params


def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]  # indexes entire row "vectorized" if state is (2, N)
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * length * np.cos(angle)
    return kinetic_energy, potential_energy


def need_reset(state, params):
    N = params['N']
    gamma = params['gamma']
    alpha = params['alpha']
    angle = state[0]

    return angle >= (gamma+alpha)


def reset_state(state, params):
    N = params['N']
    alpha = params['alpha']
    angle = state[0]
    angular_velocity = state[1]
    
    angle = angle-2*alpha
    angular_velocity = angular_velocity*np.cos(2*alpha)
    return np.array([angle,angular_velocity])
