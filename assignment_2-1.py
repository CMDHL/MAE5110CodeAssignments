import matplotlib.pyplot as plt
import numpy as np

from models import inverted_pendulum_walker as model


params = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 1.0,
    "incline": 0.06,
    "angle_of_attack": np.pi / 8,
    "ankle_torque": 0.0,
}


def compute_ankle_torque(state, params):
    angle = state[0]
    angular_velocity = state[1]
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    kp = 10.0
    kd = 2 * np.sqrt(kp)

    torque = (
        -mass * gravity * length * np.sin(angle)
        - mass * length**2 * (kp * angle + kd * angular_velocity)
    )

    min_torque = -0.1 * mass * gravity * length
    max_torque = 0.05 * mass * gravity * length

    return np.clip(torque, min_torque, max_torque)

initial_state = np.array([0.02, 0.0])

timestep = 1e-3
sim_time = 10.0

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]

    params["ankle_torque"] = compute_ankle_torque(state, params)

    next_state = (
        state
        + timestep * model.dynamics(t, state, params)
    )

    state_traj[:, step + 1] = next_state


plt.figure()

plt.plot(time_traj, state_traj[0], label="angle")
plt.plot(time_traj, state_traj[1], label="angular velocity")

plt.xlabel("Time (s)")
plt.legend()
plt.grid()

plt.show()