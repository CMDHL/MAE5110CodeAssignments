from pathlib import Path

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

OUTPUT_DIR = Path("assignment_2_results")
ROA_DATA_PATH = OUTPUT_DIR / "RoA_data.npz"
ROA_FIGURE_PATH = OUTPUT_DIR / "RoA.png"


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


def has_fallen(state, params):
    angle = state[0]
    incline = params["incline"]

    return abs(angle - incline) >= np.pi / 2


def balances_from_state(
    initial_state,
    params,
    timestep=1e-3,
    sim_time=10.0,
    angle_tolerance=1e-2,
    velocity_tolerance=1e-2,
):
    state = np.array(initial_state, dtype=float)

    n_timesteps = round(sim_time / timestep)

    for step in range(n_timesteps):
        params["ankle_torque"] = compute_ankle_torque(state, params)

        state = (
            state
            + timestep
            * model.dynamics(step * timestep, state, params)
        )

        if has_fallen(state, params):
            return False

    return (
        abs(state[0]) < angle_tolerance
        and abs(state[1]) < velocity_tolerance
    )


def compute_balance_roa(params):
    max_angle = np.pi / 2
    max_velocity = np.sqrt(
        2 * params["gravity"] / params["length"]
    )

    angle_values = np.linspace(
        params["incline"] - max_angle,
        params["incline"] + max_angle,
        101,
    )

    angular_velocity_values = np.linspace(
        -max_velocity,
        max_velocity,
        101,
    )

    roa = np.zeros(
        (
            len(angular_velocity_values),
            len(angle_values),
        ),
        dtype=bool,
    )

    for i, angular_velocity in enumerate(angular_velocity_values):
        print(
            f"RoA row {i + 1}/{len(angular_velocity_values)}",
            end="\r",
        )

        for j, angle in enumerate(angle_values):
            initial_state = np.array(
                [angle, angular_velocity]
            )

            roa[i, j] = balances_from_state(
                initial_state,
                params,
            )

    print()

    return angle_values, angular_velocity_values, roa


def save_roa_data(angle_values, angular_velocity_values, roa):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        ROA_DATA_PATH,
        angle_values=angle_values,
        angular_velocity_values=angular_velocity_values,
        roa=roa,
    )


def load_or_compute_roa(params):
    if ROA_DATA_PATH.exists():
        print(f"Loading existing RoA data from {ROA_DATA_PATH}")

        data = np.load(ROA_DATA_PATH)

        return (
            data["angle_values"],
            data["angular_velocity_values"],
            data["roa"],
        )

    print("Computing RoA...")

    angle_values, angular_velocity_values, roa = (
        compute_balance_roa(params)
    )

    save_roa_data(
        angle_values,
        angular_velocity_values,
        roa,
    )

    return angle_values, angular_velocity_values, roa


def plot_roa(angle_values, angular_velocity_values, roa):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.imshow(
        roa,
        origin="lower",
        aspect="auto",
        extent=[
            angle_values[0],
            angle_values[-1],
            angular_velocity_values[0],
            angular_velocity_values[-1],
        ],
    )

    ax.scatter(
        [0],
        [0],
        marker="x",
        label="Upright equilibrium",
    )

    ax.set_xlabel("Angle (rad)")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title("Region of Attraction of the Ankle Controller")
    ax.legend()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        ROA_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved {ROA_FIGURE_PATH}")

    plt.show()


def main():
    angle_values, angular_velocity_values, roa = (
        load_or_compute_roa(params)
    )

    plot_roa(
        angle_values,
        angular_velocity_values,
        roa,
    )


if __name__ == "__main__":
    main()