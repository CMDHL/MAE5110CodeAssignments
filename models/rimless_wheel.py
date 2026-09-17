import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping_coeff = params["damping_coeff"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        gravity / length * np.sin(angle)
        - damping_coeff * angular_velocity / (mass * length**2)
    )

    return np.array([angular_velocity, angular_acceleration])


def generate_params(N=8, gamma=0.1):
    params = {
        "N": N,  # number of spokes
        "gamma": gamma,  # downhill inclination in rad
        "gravity": 9.81,  # gravity (m/s^2)
        "length": 1.0,  # spoke length (m)
        "mass": 1.0,  # point mass at hub (kg)
        "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
    }
    params["alpha"] = np.pi / params["N"]
    return params


def calculate_energy(state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    gamma = params["gamma"]
    alpha = params["alpha"]

    angle = state[0]
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    if np.ndim(angle) == 0:
        contact_height = 0.0
    else:
        resets = np.diff(angle) < -alpha
        step_number = np.concatenate(([0], np.cumsum(resets)))
        contact_height_drop = 2 * length * np.sin(alpha) * np.sin(gamma)
        contact_height = -step_number * contact_height_drop

    hub_height = contact_height + length * np.cos(angle)
    potential_energy = mass * gravity * hub_height

    return kinetic_energy, potential_energy


def need_reset(state, params):
    angle = state[0]
    angular_velocity = state[1]
    return angle >= params["gamma"] + params["alpha"] and angular_velocity > 0


def reset_state(state, params):
    alpha = params["alpha"]
    angle = state[0]
    angular_velocity = state[1]

    angle = angle - 2 * alpha
    angular_velocity = angular_velocity * np.cos(2 * alpha)
    return np.array([angle, angular_velocity])
