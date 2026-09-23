import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models import inverted_pendulum_walker as model


params = model.generate_params()

OUTPUT_DIR = Path("assignment_2_results")
ROA_DATA_PATH = OUTPUT_DIR / "RoA_data.npz"
POINCARE_DATA_PATH = OUTPUT_DIR / "poincare_table.npz"
RETURN_MAP_FIGURE_PATH = OUTPUT_DIR / "poincare_return_map.png"
GRID_RESOLUTION_PATH = OUTPUT_DIR / "grid_resolution.csv"

MIN_ANGLE_OF_ATTACK = np.pi / 8
MAX_ANGLE_OF_ATTACK = np.pi / 7
STATE_ERROR_TOLERANCE = 1e-2
CONTROL_STEP_TOLERANCE = 1.5e-3


def calculate_next_section_speed(angular_velocity, angle_of_attack, params):
    gravity = params["gravity"]
    length = params["length"]
    incline = params["incline"]

    touchdown_speed_squared = (
        angular_velocity**2
        + 2
        * gravity
        / length
        * (
            np.cos(incline)
            - np.cos(incline + angle_of_attack)
        )
    )

    if touchdown_speed_squared < 0:
        return np.nan

    post_impact_speed_squared = (
        np.cos(2 * angle_of_attack) ** 2
        * touchdown_speed_squared
    )

    next_speed_squared = (
        post_impact_speed_squared
        + 2
        * gravity
        / length
        * (
            np.cos(incline - angle_of_attack)
            - np.cos(incline)
        )
    )

    if next_speed_squared < 0:
        return np.nan

    return np.sqrt(next_speed_squared)


def load_roa_upper_speed(params):
    data = np.load(ROA_DATA_PATH)
    angle_values = data["angle_values"]
    angular_velocity_values = data["angular_velocity_values"]
    roa = data["roa"]

    angle_index = np.argmin(abs(angle_values - params["incline"]))
    section_roa = roa[:, angle_index]
    positive_roa = section_roa & (angular_velocity_values >= 0)

    return np.max(angular_velocity_values[positive_roa])


def build_poincare_table(params, speed_count, control_count):
    max_speed = np.sqrt(2 * params["gravity"] / params["length"])

    speed_values = np.linspace(0, max_speed, speed_count)
    angle_of_attack_values = np.linspace(
        MIN_ANGLE_OF_ATTACK,
        MAX_ANGLE_OF_ATTACK,
        control_count,
    )

    next_speed_grid = np.zeros(
        (
            len(speed_values),
            len(angle_of_attack_values),
        )
    )

    for i, angular_velocity in enumerate(speed_values):
        for j, angle_of_attack in enumerate(angle_of_attack_values):
            next_speed_grid[i, j] = calculate_next_section_speed(
                angular_velocity,
                angle_of_attack,
                params,
            )

    return speed_values, angle_of_attack_values, next_speed_grid


def calculate_rounding_error(speed_values, next_speed_grid):
    rounding_error = 0.0

    for next_speed in next_speed_grid[np.isfinite(next_speed_grid)]:
        speed_index = np.argmin(abs(speed_values - next_speed))
        error = abs(speed_values[speed_index] - next_speed)
        rounding_error = max(rounding_error, error)

    return rounding_error


def evaluate_grid_resolution(params):
    rows = []

    for speed_count, control_count in [
        (222, 39),
        (223, 38),
        (223, 39),
    ]:
        speed_values, angle_of_attack_values, next_speed_grid = (
            build_poincare_table(
                params,
                speed_count,
                control_count,
            )
        )

        speed_spacing = speed_values[1] - speed_values[0]
        control_spacing = (
            angle_of_attack_values[1]
            - angle_of_attack_values[0]
        )
        rounding_error = calculate_rounding_error(
            speed_values,
            next_speed_grid,
        )
        passes = (
            rounding_error <= STATE_ERROR_TOLERANCE
            and control_spacing <= CONTROL_STEP_TOLERANCE
        )

        rows.append(
            {
                "speed_count": speed_count,
                "control_count": control_count,
                "speed_spacing": speed_spacing,
                "control_spacing": control_spacing,
                "max_rounding_error": rounding_error,
                "passes": passes,
            }
        )

    return rows


def save_resolution_table(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with GRID_RESOLUTION_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_poincare_table(
    speed_values,
    angle_of_attack_values,
    next_speed_grid,
    roa_upper_speed,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        POINCARE_DATA_PATH,
        speed_values=speed_values,
        angle_of_attack_values=angle_of_attack_values,
        next_speed_grid=next_speed_grid,
        roa_upper_speed=roa_upper_speed,
        section_angle=params["incline"],
    )


def plot_return_map(
    speed_values,
    angle_of_attack_values,
    next_speed_grid,
    roa_upper_speed,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for angle_of_attack in [
        angle_of_attack_values[0],
        angle_of_attack_values[len(angle_of_attack_values) // 2],
        angle_of_attack_values[-1],
    ]:
        control_index = np.argmin(
            abs(angle_of_attack_values - angle_of_attack)
        )
        ax.plot(
            speed_values,
            next_speed_grid[:, control_index],
            label=(
                r"$\alpha$ = "
                f"{angle_of_attack_values[control_index]:.3f} rad"
            ),
        )

    ax.plot(
        speed_values,
        speed_values,
        "k--",
        label="identity",
    )
    ax.axhspan(
        0,
        roa_upper_speed,
        color="tab:green",
        alpha=0.18,
        label="RoA at section",
    )
    ax.set_xlabel(r"Current section speed $\dot{\theta}_k$ (rad/s)")
    ax.set_ylabel(r"Next section speed $\dot{\theta}_{k+1}$ (rad/s)")
    ax.set_title(r"Poincare map at $\theta = \gamma$")
    ax.legend()
    ax.grid(alpha=0.25)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        RETURN_MAP_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved {RETURN_MAP_FIGURE_PATH}")


def main():
    rows = evaluate_grid_resolution(params)
    save_resolution_table(rows)

    speed_values, angle_of_attack_values, next_speed_grid = (
        build_poincare_table(
            params,
            speed_count=223,
            control_count=39,
        )
    )
    roa_upper_speed = load_roa_upper_speed(params)

    save_poincare_table(
        speed_values,
        angle_of_attack_values,
        next_speed_grid,
        roa_upper_speed,
    )
    plot_return_map(
        speed_values,
        angle_of_attack_values,
        next_speed_grid,
        roa_upper_speed,
    )

    print(f"Saved {POINCARE_DATA_PATH}")
    print(f"Saved {GRID_RESOLUTION_PATH}")
    print(f"RoA upper speed at section: {roa_upper_speed:.6f} rad/s")


if __name__ == "__main__":
    main()
