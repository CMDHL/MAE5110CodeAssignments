import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from integrators import rk4 as integrator
from models import rimless_wheel as model

RESULTS = Path("assignment_1_results")
STANDING = 0
ROLLING = 1
UNRESOLVED = 2


def simulate(initial_state, timestep, sim_time, params):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    impact_speeds = []
    state_traj[:, 0] = initial_state

    for i, t in enumerate(time_traj[:-1]):
        next_state = integrator.step(t, state_traj[:, i], timestep, params, model)
        if model.need_reset(next_state, params):
            next_state = model.reset_state(next_state, params)
            impact_speeds.append(next_state[1])
        state_traj[:, i + 1] = next_state

    return time_traj, state_traj, np.array(impact_speeds)


def get_step_angles(params):
    gamma = params["gamma"]
    alpha = params["alpha"]
    return gamma - alpha, gamma + alpha


def get_speed_gain(params):
    gravity = params["gravity"]
    length = params["length"]
    lower_angle, upper_angle = get_step_angles(params)
    return 2 * gravity / length * (np.cos(lower_angle) - np.cos(upper_angle))


def get_forward_threshold(params):
    gravity = params["gravity"]
    length = params["length"]
    lower_angle, upper_angle = get_step_angles(params)
    if lower_angle <= 0 <= upper_angle:
        peak = 1.0
    else:
        peak = max(np.cos(lower_angle), np.cos(upper_angle))
    return np.sqrt(max(0, 2 * gravity / length * (peak - np.cos(lower_angle))))


def get_backward_threshold(params):
    gravity = params["gravity"]
    length = params["length"]
    lower_angle, upper_angle = get_step_angles(params)
    if lower_angle <= 0 <= upper_angle:
        peak = 1.0
    else:
        peak = max(np.cos(lower_angle), np.cos(upper_angle))
    speed = np.sqrt(max(0, 2 * gravity / length * (peak - np.cos(upper_angle))))
    return -speed


def calculate_return_map(post_impact_speed, params):
    speeds = np.atleast_1d(post_impact_speed).astype(float)
    mapped_speeds = np.full_like(speeds, np.nan)
    collision_loss = np.cos(2 * params["alpha"])
    speed_gain = get_speed_gain(params)
    forward_threshold = get_forward_threshold(params)
    backward_threshold = get_backward_threshold(params)

    rolling_forward = speeds > forward_threshold
    rocking = (speeds > backward_threshold) & (speeds < forward_threshold)
    rolling_backward = speeds < backward_threshold

    mapped_speeds[rolling_forward] = collision_loss * np.sqrt(
        speeds[rolling_forward] ** 2 + speed_gain
    )
    mapped_speeds[rocking] = -collision_loss * speeds[rocking]
    mapped_speeds[rolling_backward] = -collision_loss * np.sqrt(
        speeds[rolling_backward] ** 2 - speed_gain
    )

    if np.ndim(post_impact_speed) == 0:
        return mapped_speeds[0]
    return mapped_speeds.reshape(np.shape(post_impact_speed))


def calculate_fixed_speed(params):
    collision_loss = np.cos(2 * params["alpha"])
    speed_gain = get_speed_gain(params)
    denominator = 1 - collision_loss**2

    if collision_loss <= 0 or speed_gain <= 0 or denominator <= 0:
        return np.nan

    fixed_speed = collision_loss * np.sqrt(speed_gain / denominator)
    if fixed_speed <= get_forward_threshold(params):
        return np.nan
    return fixed_speed


def calculate_floquet_multiplier(params):
    if np.isnan(calculate_fixed_speed(params)):
        return np.nan
    return np.cos(2 * params["alpha"]) ** 2


def get_first_section_speed(angle, speed, params):
    gravity = params["gravity"]
    length = params["length"]
    lower_angle, upper_angle = get_step_angles(params)
    collision_loss = np.cos(2 * params["alpha"])
    energy = 0.5 * speed**2 + gravity / length * np.cos(angle)
    upright_energy = gravity / length

    moving_forward = (
        (speed > 0) & ((angle >= 0) | (energy > upright_energy))
    ) | ((speed < 0) & (angle > 0) & (energy < upright_energy))
    moving_backward = (
        (speed < 0) & ((angle <= 0) | (energy > upright_energy))
    ) | ((speed > 0) & (angle < 0) & (energy < upright_energy))
    moving_forward |= (speed == 0) & (angle > 0)
    moving_backward |= (speed == 0) & (angle < 0)

    section_speed = np.full_like(speed, np.nan)
    forward_speed = speed**2 + 2 * gravity / length * (
        np.cos(angle) - np.cos(upper_angle)
    )
    backward_speed = speed**2 + 2 * gravity / length * (
        np.cos(angle) - np.cos(lower_angle)
    )
    section_speed[moving_forward] = collision_loss * np.sqrt(
        np.maximum(forward_speed[moving_forward], 0)
    )
    section_speed[moving_backward] = -collision_loss * np.sqrt(
        np.maximum(backward_speed[moving_backward], 0)
    )
    return section_speed


def estimate_roa(params, angle_count=101, speed_count=151, max_steps=150):
    lower_angle, upper_angle = get_step_angles(params)
    speed_limit = 1.25 * np.sqrt(params["gravity"] / params["length"])
    angles = np.linspace(lower_angle, upper_angle, angle_count)
    speeds = np.linspace(-speed_limit, speed_limit, speed_count)
    angle_grid, speed_grid = np.meshgrid(angles, speeds)
    section_speed = get_first_section_speed(angle_grid, speed_grid, params)

    attractors = np.full(section_speed.shape, UNRESOLVED)
    active = np.isfinite(section_speed)
    speed_tolerance = 1e-3 * np.sqrt(params["gravity"] / params["length"])
    fixed_speed = calculate_fixed_speed(params)

    for _ in range(max_steps):
        standing = active & (np.abs(section_speed) < speed_tolerance)
        attractors[standing] = STANDING
        active[standing] = False

        if np.isfinite(fixed_speed):
            rolling = active & (np.abs(section_speed - fixed_speed) < speed_tolerance)
            attractors[rolling] = ROLLING
            active[rolling] = False

        if not np.any(active):
            break

        active_index = np.flatnonzero(active)
        section_speed.flat[active_index] = calculate_return_map(
            section_speed.flat[active_index], params
        )
        active = active & np.isfinite(section_speed)

    return angles, speeds, attractors


def save_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_sanity(args):
    params = model.generate_params()
    lower_angle, _ = get_step_angles(params)
    time_traj, state_traj, impact_speeds = simulate(
        np.array([lower_angle, 1.5]), args.timestep, 10.0, params
    )
    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)
    total_energy = kinetic_energy + potential_energy

    RESULTS.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(time_traj, state_traj[0], label="angle")
    axes[0, 0].plot(time_traj, state_traj[1], label="angular velocity")
    axes[0, 0].set(xlabel="time (s)", title="state trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(state_traj[0], state_traj[1])
    axes[0, 1].set(xlabel="angle (rad)", ylabel="angular velocity (rad/s)")
    axes[0, 1].set_title("phase portrait")

    axes[1, 0].plot(time_traj, kinetic_energy, label="kinetic")
    axes[1, 0].plot(time_traj, potential_energy, label="potential")
    axes[1, 0].plot(time_traj, total_energy, label="total")
    axes[1, 0].set(xlabel="time (s)", ylabel="energy (J)")
    axes[1, 0].legend()

    axes[1, 1].plot(np.arange(1, impact_speeds.size + 1), impact_speeds, "o-")
    axes[1, 1].axhline(calculate_fixed_speed(params), color="black", ls="--")
    axes[1, 1].set(xlabel="step number", ylabel="post-impact speed (rad/s)")
    axes[1, 1].set_title("speed approaches the limit cycle")

    figure.tight_layout()
    output_path = RESULTS / "sanity_checks.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"saved {output_path}")
    print(f"last post-impact speed: {impact_speeds[-1]:.6f} rad/s")


def run_roa(args):
    params = model.generate_params()
    angles, speeds, attractors = estimate_roa(
        params, angle_count=args.angle_count, speed_count=args.speed_count
    )
    rolling_fraction = np.mean(attractors == ROLLING)
    standing_fraction = np.mean(attractors == STANDING)

    RESULTS.mkdir(exist_ok=True)
    angle_grid, speed_grid = np.meshgrid(angles, speeds)
    color_map = ListedColormap(["#4c78a8", "#f58518", "#d9d9d9"])
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.pcolormesh(
        angle_grid,
        speed_grid,
        attractors,
        shading="nearest",
        cmap=color_map,
        vmin=-0.5,
        vmax=2.5,
    )
    axis.scatter([0], [0], marker="x", color="black", label="upright separatrix")
    axis.set(
        xlabel="initial angle (rad)",
        ylabel="initial angular velocity (rad/s)",
        title="regions of attraction",
    )
    patches = [
        Patch(color="#4c78a8", label="standing"),
        Patch(color="#f58518", label="rolling"),
        Patch(color="#d9d9d9", label="unresolved"),
    ]
    axis.legend(handles=patches + axis.get_legend_handles_labels()[0])
    figure.tight_layout()
    output_path = RESULTS / "region_of_attraction.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"saved {output_path}")
    print(f"rolling: {100 * rolling_fraction:.2f}%")
    print(f"standing: {100 * standing_fraction:.2f}%")


def run_return_map(args):
    params = model.generate_params()
    speeds = np.linspace(-2.4, 3.0, 400)
    mapped_speeds = calculate_return_map(speeds, params)
    fixed_speed = calculate_fixed_speed(params)

    RESULTS.mkdir(exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 6))
    axis.plot(speeds, mapped_speeds, label="return map")
    axis.plot(speeds, speeds, "k--", label="identity")
    axis.scatter([0, fixed_speed], [0, fixed_speed], color="black", zorder=3)
    axis.axvline(get_forward_threshold(params), color="0.5", ls=":")
    axis.axvline(get_backward_threshold(params), color="0.5", ls=":")
    axis.set(
        xlabel="current post-impact speed (rad/s)",
        ylabel="next post-impact speed (rad/s)",
        title="step-to-step return map",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = RESULTS / "return_map.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"saved {output_path}")
    print(f"rolling fixed point: {fixed_speed:.6f} rad/s")


def run_floquet(args):
    params = model.generate_params()
    fixed_speed = calculate_fixed_speed(params)
    delta = args.perturbation * fixed_speed
    left_speed = fixed_speed - delta
    right_speed = fixed_speed + delta
    left_slope = (
        calculate_return_map(fixed_speed, params)
        - calculate_return_map(left_speed, params)
    ) / delta
    right_slope = (
        calculate_return_map(right_speed, params)
        - calculate_return_map(fixed_speed, params)
    ) / delta
    centered_slope = (
        calculate_return_map(right_speed, params)
        - calculate_return_map(left_speed, params)
    ) / (2 * delta)

    RESULTS.mkdir(exist_ok=True)
    output_path = RESULTS / "floquet.json"
    values = {
        "fixed_speed_rad_per_s": fixed_speed,
        "left_slope": left_slope,
        "right_slope": right_slope,
        "centered_floquet_multiplier": centered_slope,
        "exact_floquet_multiplier": calculate_floquet_multiplier(params),
    }
    with output_path.open("w") as file:
        json.dump(values, file, indent=2)
        file.write("\n")

    print(f"saved {output_path}")
    print(f"centered Floquet multiplier: {centered_slope:.6f}")


def run_slope_sweep(args):
    rows = []
    for gamma in np.arange(0.04, 0.361, 0.04):
        params = model.generate_params(gamma=float(gamma))
        _, _, attractors = estimate_roa(
            params, angle_count=args.angle_count, speed_count=args.speed_count
        )
        rows.append(
            {
                "gamma_rad": gamma,
                "rolling_fraction": np.mean(attractors == ROLLING),
                "fixed_speed_rad_per_s": calculate_fixed_speed(params),
                "floquet_multiplier": calculate_floquet_multiplier(params),
            }
        )

    RESULTS.mkdir(exist_ok=True)
    save_csv(RESULTS / "slope_sweep.csv", rows)
    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    gamma = [row["gamma_rad"] for row in rows]
    axes[0].plot(gamma, [row["rolling_fraction"] for row in rows], "o-")
    axes[0].set(ylabel="rolling fraction", title="slope sweep")
    axes[1].plot(gamma, [row["floquet_multiplier"] for row in rows], "o-")
    axes[1].set(xlabel="slope gamma (rad)", ylabel="Floquet multiplier")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = RESULTS / "slope_sweep.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"saved {output_path}")
    print(f"saved {RESULTS / 'slope_sweep.csv'}")


def run_spoke_sweep(args):
    rows = []
    for N in range(6, 13):
        params = model.generate_params(N=N, gamma=0.2)
        _, _, attractors = estimate_roa(
            params, angle_count=args.angle_count, speed_count=args.speed_count
        )
        rows.append(
            {
                "N": N,
                "rolling_fraction": np.mean(attractors == ROLLING),
                "fixed_speed_rad_per_s": calculate_fixed_speed(params),
                "floquet_multiplier": calculate_floquet_multiplier(params),
            }
        )

    RESULTS.mkdir(exist_ok=True)
    save_csv(RESULTS / "spoke_sweep.csv", rows)
    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    spokes = [row["N"] for row in rows]
    axes[0].plot(spokes, [row["rolling_fraction"] for row in rows], "o-")
    axes[0].set(ylabel="rolling fraction", title="spoke sweep")
    axes[1].plot(spokes, [row["floquet_multiplier"] for row in rows], "o-")
    axes[1].set(xlabel="number of spokes", ylabel="Floquet multiplier")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path = RESULTS / "spoke_sweep.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"saved {output_path}")
    print(f"saved {RESULTS / 'spoke_sweep.csv'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=[
            "sanity",
            "roa",
            "return-map",
            "floquet",
            "slope-sweep",
            "spoke-sweep",
        ],
    )
    parser.add_argument("--timestep", type=float, default=1e-3)
    parser.add_argument("--angle-count", type=int, default=101)
    parser.add_argument("--speed-count", type=int, default=151)
    parser.add_argument("--perturbation", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    runners = {
        "sanity": run_sanity,
        "roa": run_roa,
        "return-map": run_return_map,
        "floquet": run_floquet,
        "slope-sweep": run_slope_sweep,
        "spoke-sweep": run_spoke_sweep,
    }
    runners[args.task](args)


if __name__ == "__main__":
    main()
