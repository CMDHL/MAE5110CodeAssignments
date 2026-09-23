import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from models import inverted_pendulum_walker as model


params = model.generate_params()

OUTPUT_DIR = Path("assignment_2_results")
POINCARE_DATA_PATH = OUTPUT_DIR / "poincare_table.npz"
POLICY_DATA_PATH = OUTPUT_DIR / "step_policy_data.npz"
GRID_RESOLUTION_PATH = OUTPUT_DIR / "grid_resolution.csv"
SKETCH_FIGURE_PATH = OUTPUT_DIR / "walker_sketches.png"
STEPS_FIGURE_PATH = OUTPUT_DIR / "steps_to_standstill.png"
TRAJECTORY_FIGURE_PATH = OUTPUT_DIR / "trajectory_examples.png"
REPORT_PATH = Path("assignment_2_report.md")
REPORT_PDF_PATH = OUTPUT_DIR / "assignment_2_report.pdf"

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


def read_resolution_rows():
    with GRID_RESOLUTION_PATH.open() as file:
        return list(csv.DictReader(file))


def make_markdown_table(rows):
    lines = [
        "| speed count | control count | speed spacing | control spacing | max rounding error | passes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['speed_count']} | "
            f"{row['control_count']} | "
            f"{float(row['speed_spacing']):.6f} | "
            f"{float(row['control_spacing']):.6f} | "
            f"{float(row['max_rounding_error']):.6f} | "
            f"{row['passes']} |"
        )

    return "\n".join(lines)


def write_report(
    rows,
    initial_speed,
    min_speed_trace,
    min_action_trace,
    max_speed_trace,
    max_action_trace,
    roa_upper_speed,
):
    report = f"""# Assignment 2 Report

## Sketches

The sketches show the mid-stance Poincare section, two touchdown guards for the allowed angle-of-attack limits, and a small state inside the balancing controller's region of attraction.

![Walker sketches]({SKETCH_FIGURE_PATH})

## 1. Feedback linearization and RoA

The ankle controller cancels the pendulum gravity term and then applies a PD term toward upright standing. The torque is clipped to the required interval [-0.1 m g l, 0.05 m g l]. Outside the RoA, the walking lookup table keeps ankle torque at zero.

![Region of attraction](assignment_2_results/RoA.png)

On the chosen section, the positive-speed edge of the RoA is {roa_upper_speed:.6f} rad/s.

## 2. Poincare section

I used the mid-stance section theta = gamma. This section is transverse for the forward walking states because angular velocity is positive when the walker crosses it, and theta is fixed, so the return map only needs the state theta_dot_k. The step controller chooses alpha once per stance phase.

![Poincare return map]({OUTPUT_DIR / "poincare_return_map.png"})

## 3. Lookup table

The state grid covers theta_dot in [0, sqrt(2 g / l)]. The control grid covers alpha in [pi/8, pi/7]. I used the coarsest grid from the table below that satisfied both checks: nearest-neighbor state rounding error at most {1e-2:.3f} rad/s, and alpha spacing at most {1.5e-3:.4f} rad.

{make_markdown_table(rows)}

The 201 by 31 grid is not good enough because both the state rounding error and alpha spacing miss the criteria. The 251 by 31 grid fixes the state spacing, but the control spacing is still too coarse. The 251 by 41 grid is the first tested grid that passes both.

![Steps to standstill]({STEPS_FIGURE_PATH})

For the trajectory example, I used initial section speed {initial_speed:.6f} rad/s. The fewest-step policy reaches the RoA in {len(min_action_trace)} steps, with alpha values {np.array2string(min_action_trace, precision=4)}. The longest successful policy reaches the RoA in {len(max_action_trace)} steps, with alpha values {np.array2string(max_action_trace, precision=4)}.

![Trajectory examples]({TRAJECTORY_FIGURE_PATH})
"""

    REPORT_PATH.write_text(report)

    print(f"Saved {REPORT_PATH}")


def add_text_page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(
        0.08,
        0.94,
        title,
        fontsize=18,
        weight="bold",
        va="top",
    )

    y = 0.88

    for paragraph in paragraphs:
        for line in textwrap.wrap(paragraph, width=92):
            fig.text(
                0.08,
                y,
                line,
                fontsize=10,
                va="top",
            )
            y -= 0.026

        y -= 0.014

    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf, title, image_path):
    image = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(8.5, 11))

    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=16)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(
    rows,
    initial_speed,
    min_action_trace,
    max_action_trace,
    roa_upper_speed,
):
    row_text = [
        (
            f"{row['speed_count']} x {row['control_count']}: "
            f"rounding error {float(row['max_rounding_error']):.6f}, "
            f"alpha spacing {float(row['control_spacing']):.6f}, "
            f"passes = {row['passes']}"
        )
        for row in rows
    ]

    with PdfPages(REPORT_PDF_PATH) as pdf:
        add_text_page(
            pdf,
            "Assignment 2 Report",
            [
                "The analysis is split across assignment_2-1.py, assignment_2-2.py, and assignment_2-3.py.",
                "Section 1 builds the feedback-linearized ankle controller and RoA. Section 2 builds the Poincare return map at theta = gamma. Section 3 backs out the lookup policy and trajectory examples.",
                f"The positive-speed RoA edge at the section is {roa_upper_speed:.6f} rad/s.",
                f"The trajectory example starts at {initial_speed:.6f} rad/s. The fewest-step policy uses {len(min_action_trace)} steps, and the longest successful policy uses {len(max_action_trace)} steps.",
            ],
        )
        add_image_page(pdf, "Sketches", SKETCH_FIGURE_PATH)
        add_image_page(pdf, "Region of Attraction", OUTPUT_DIR / "RoA.png")
        add_image_page(pdf, "Poincare Return Map", OUTPUT_DIR / "poincare_return_map.png")
        add_text_page(
            pdf,
            "Grid Resolution",
            [
                "Criterion: nearest-neighbor state rounding error <= 0.010 rad/s and alpha spacing <= 0.0015 rad.",
                *row_text,
            ],
        )
        add_image_page(pdf, "Steps to Standstill", STEPS_FIGURE_PATH)
        add_image_page(pdf, "Trajectory Examples", TRAJECTORY_FIGURE_PATH)

    print(f"Saved {REPORT_PDF_PATH}")


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

    rows = read_resolution_rows()

    write_report(
        rows,
        initial_speed,
        min_speed_trace,
        min_action_trace,
        max_speed_trace,
        max_action_trace,
        roa_upper_speed,
    )
    write_pdf_report(
        rows,
        initial_speed,
        min_action_trace,
        max_action_trace,
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
