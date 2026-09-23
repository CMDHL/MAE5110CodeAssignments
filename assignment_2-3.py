from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models import inverted_pendulum_walker as model


params = model.generate_params()

OUTPUT_DIR = Path("assignment_2_results")
POINCARE_DATA_PATH = OUTPUT_DIR / "poincare_table.npz"
POLICY_DATA_PATH = OUTPUT_DIR / "step_policy_data.npz"
SKETCH_FIGURE_PATH = OUTPUT_DIR / "walker_sketches.png"
STEPS_FIGURE_PATH = OUTPUT_DIR / "steps_to_standstill.png"
TRAJECTORY_FIGURE_PATH = OUTPUT_DIR / "trajectory_examples.png"

MAX_STEPS = 20


def get_nearest_index(values, value):
    return np.argmin(abs(values - value))


def build_step_policy(
    speed_values,
    angle_of_attack_values,
    next_speed_grid,
    roa_upper_speed,
):
    step_counts = np.full(len(speed_values), -1)
    best_actions = np.full(len(speed_values), np.nan)
    max_step_counts = np.full(len(speed_values), -1)
    longest_actions = np.full(len(speed_values), np.nan)

    already_in_roa = speed_values <= roa_upper_speed
    step_counts[already_in_roa] = 0
    max_step_counts[already_in_roa] = 0

    for _ in range(MAX_STEPS):
        previous_counts = step_counts.copy()
        changed = False

        for i, _ in enumerate(speed_values):
            if previous_counts[i] >= 0:
                continue

            best_step_count = MAX_STEPS + 1

            for j, next_speed in enumerate(next_speed_grid[i]):
                if not np.isfinite(next_speed):
                    continue

                next_index = get_nearest_index(
                    speed_values,
                    next_speed,
                )

                if previous_counts[next_index] < 0:
                    continue

                candidate_count = previous_counts[next_index] + 1

                if candidate_count < best_step_count:
                    best_step_count = candidate_count
                    best_actions[i] = angle_of_attack_values[j]

            if best_step_count <= MAX_STEPS:
                step_counts[i] = best_step_count
                changed = True

        if not changed:
            break

    for _ in range(MAX_STEPS):
        previous_counts = max_step_counts.copy()
        changed = False

        for i, current_speed in enumerate(speed_values):
            if already_in_roa[i]:
                continue

            best_step_count = max_step_counts[i]

            for j, next_speed in enumerate(next_speed_grid[i]):
                if not np.isfinite(next_speed):
                    continue

                next_index = get_nearest_index(
                    speed_values,
                    next_speed,
                )

                if previous_counts[next_index] < 0:
                    continue
                if speed_values[next_index] >= current_speed:
                    continue

                candidate_count = previous_counts[next_index] + 1

                if candidate_count > best_step_count:
                    best_step_count = candidate_count
                    longest_actions[i] = angle_of_attack_values[j]

            if best_step_count > max_step_counts[i]:
                max_step_counts[i] = best_step_count
                changed = True

        if not changed:
            break

    return (
        step_counts,
        best_actions,
        max_step_counts,
        longest_actions,
    )


def save_policy_data(
    speed_values,
    step_counts,
    best_actions,
    max_step_counts,
    longest_actions,
    initial_speed,
):
    np.savez(
        POLICY_DATA_PATH,
        speed_values=speed_values,
        step_counts=step_counts,
        best_actions=best_actions,
        max_step_counts=max_step_counts,
        longest_actions=longest_actions,
        initial_speed=initial_speed,
    )


def choose_initial_speed(speed_values, step_counts, max_step_counts):
    candidates = np.flatnonzero(
        (step_counts >= 3)
        & (max_step_counts >= step_counts)
    )

    if len(candidates) == 0:
        candidates = np.flatnonzero(step_counts >= 0)

    best_index = candidates[
        np.argmax(max_step_counts[candidates] - step_counts[candidates])
    ]

    return speed_values[best_index]


def trace_policy(
    initial_speed,
    speed_values,
    angle_of_attack_values,
    next_speed_grid,
    actions,
    roa_upper_speed,
):
    speed_trace = [initial_speed]
    action_trace = []
    current_speed = initial_speed

    for _ in range(MAX_STEPS):
        if current_speed <= roa_upper_speed:
            break

        speed_index = get_nearest_index(speed_values, current_speed)
        angle_of_attack = actions[speed_index]

        if not np.isfinite(angle_of_attack):
            break

        control_index = get_nearest_index(
            angle_of_attack_values,
            angle_of_attack,
        )
        next_speed = next_speed_grid[speed_index, control_index]

        action_trace.append(angle_of_attack)
        speed_trace.append(next_speed)
        current_speed = next_speed

    return np.array(speed_trace), np.array(action_trace)


def plot_walker_sketches():
    incline = params["incline"]

    states = [
        np.array([incline, 2.0]),
        np.array([incline + np.pi / 8, 2.0]),
        np.array([incline + np.pi / 7, 2.0]),
        np.array([0.02, 0.04]),
    ]
    titles = [
        r"section: $\theta = \gamma$",
        r"touchdown with $\alpha = \pi/8$",
        r"touchdown with $\alpha = \pi/7$",
        "inside balancing RoA",
    ]
    angles = [
        (np.pi / 8 + np.pi / 7) / 2,
        np.pi / 8,
        np.pi / 7,
        np.pi / 8,
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))

    for i, ax in enumerate(axes.flat):
        params["angle_of_attack"] = angles[i]
        model.visualize(
            states[i],
            params,
            ax=ax,
            show_swing=i != 3,
            view_limits=(-1.4, 1.4, -0.5, 1.5),
        )
        ax.set_title(titles[i])

    fig.tight_layout()
    fig.savefig(
        SKETCH_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved {SKETCH_FIGURE_PATH}")


def plot_step_counts(speed_values, step_counts, roa_upper_speed):
    fig, ax = plt.subplots(figsize=(8, 5))

    reachable = step_counts >= 0
    unreachable = ~reachable

    scatter = ax.scatter(
        speed_values[reachable],
        step_counts[reachable],
        c=step_counts[reachable],
        cmap="viridis",
        s=18,
    )
    ax.scatter(
        speed_values[unreachable],
        np.zeros(np.sum(unreachable)),
        color="0.7",
        marker="x",
        label="not found",
    )
    ax.axvspan(
        0,
        roa_upper_speed,
        color="tab:green",
        alpha=0.18,
        label="already in RoA",
    )
    ax.set_xlabel(r"Initial section speed $\dot{\theta}_0$ (rad/s)")
    ax.set_ylabel("Minimum steps to RoA")
    ax.set_title("Lookup table steps to standstill")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="steps")

    fig.savefig(
        STEPS_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved {STEPS_FIGURE_PATH}")


def plot_trajectories(
    initial_speed,
    min_speed_trace,
    max_speed_trace,
    roa_upper_speed,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        np.arange(len(min_speed_trace)),
        min_speed_trace,
        "o-",
        label="fewest-step policy",
    )
    ax.plot(
        np.arange(len(max_speed_trace)),
        max_speed_trace,
        "s-",
        label="longest successful policy",
    )
    ax.axhspan(
        0,
        roa_upper_speed,
        color="tab:green",
        alpha=0.18,
        label="RoA at section",
    )
    ax.set_xlabel("Step number")
    ax.set_ylabel(r"Section speed $\dot{\theta}$ (rad/s)")
    ax.set_title(
        "Trajectory from "
        f"{initial_speed:.3f} rad/s"
    )
    ax.legend()
    ax.grid(alpha=0.25)

    fig.savefig(
        TRAJECTORY_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved {TRAJECTORY_FIGURE_PATH}")


def main():
    data = np.load(POINCARE_DATA_PATH)
    speed_values = data["speed_values"]
    angle_of_attack_values = data["angle_of_attack_values"]
    next_speed_grid = data["next_speed_grid"]
    roa_upper_speed = data["roa_upper_speed"]

    (
        step_counts,
        best_actions,
        max_step_counts,
        longest_actions,
    ) = build_step_policy(
        speed_values,
        angle_of_attack_values,
        next_speed_grid,
        roa_upper_speed,
    )

    initial_speed = choose_initial_speed(
        speed_values,
        step_counts,
        max_step_counts,
    )
    min_speed_trace, min_action_trace = trace_policy(
        initial_speed,
        speed_values,
        angle_of_attack_values,
        next_speed_grid,
        best_actions,
        roa_upper_speed,
    )
    max_speed_trace, max_action_trace = trace_policy(
        initial_speed,
        speed_values,
        angle_of_attack_values,
        next_speed_grid,
        longest_actions,
        roa_upper_speed,
    )

    save_policy_data(
        speed_values,
        step_counts,
        best_actions,
        max_step_counts,
        longest_actions,
        initial_speed,
    )
    plot_walker_sketches()
    plot_step_counts(
        speed_values,
        step_counts,
        roa_upper_speed,
    )
    plot_trajectories(
        initial_speed,
        min_speed_trace,
        max_speed_trace,
        roa_upper_speed,
    )

    print(f"Saved {POLICY_DATA_PATH}")
    print(
        "Initial condition: "
        f"{initial_speed:.6f} rad/s, "
        f"{len(min_action_trace)} steps minimum, "
        f"{len(max_action_trace)} steps maximum"
    )


if __name__ == "__main__":
    main()
