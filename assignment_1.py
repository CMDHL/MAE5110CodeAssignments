"""Complete, reproducible analysis of the passive rimless wheel.

Run ``uv run python assignment_1.py`` from the repository root to regenerate
the figures and numeric tables used by ``assignment_1_report.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Arc, Patch

from integrators import rk4
from models import rimless_wheel as wheel

STANDING = 0
ROLLING = 1
UNRESOLVED = 2


@dataclass
class HybridSimulation:
    """Uniformly sampled trajectory plus localized collision states."""

    time: np.ndarray
    state: np.ndarray
    impact_times: np.ndarray
    impact_directions: np.ndarray
    pre_impact_states: np.ndarray
    post_impact_states: np.ndarray
    outcome: str


@dataclass
class StepSimulation:
    """One successful downhill post-impact-to-post-impact step."""

    time: np.ndarray
    state: np.ndarray
    impact_time: float
    pre_impact_state: np.ndarray
    post_impact_state: np.ndarray


@dataclass
class BasinEstimate:
    """Attractor labels on a rectangular state-space grid."""

    angles: np.ndarray
    angular_velocities: np.ndarray
    attractors: np.ndarray

    @property
    def rolling_fraction(self) -> float:
        """Fraction of the declared finite grid attracted to rolling."""
        return float(np.mean(self.attractors == ROLLING))

    @property
    def unresolved_fraction(self) -> float:
        """Fraction ending exactly on a separatrix or iteration limit."""
        return float(np.mean(self.attractors == UNRESOLVED))


def integrate_step(
    time: float,
    state: np.ndarray,
    timestep: float,
    params: dict[str, float],
) -> np.ndarray:
    """Advance the single-support flow by one fourth-order Runge--Kutta step."""
    return rk4.step(time, state, timestep, params, wheel)


def locate_boundary_crossing(
    time: float,
    state: np.ndarray,
    timestep: float,
    boundary_angle: float,
    direction: int,
    params: dict[str, float],
    tolerance: float = 1e-12,
) -> tuple[float, np.ndarray]:
    """Locate an angle crossing inside one RK4 step by bisection."""
    lower_offset = 0.0
    upper_offset = timestep
    while upper_offset - lower_offset > tolerance:
        midpoint = 0.5 * (lower_offset + upper_offset)
        midpoint_state = integrate_step(time, state, midpoint, params)
        if direction * (midpoint_state[0] - boundary_angle) >= 0.0:
            upper_offset = midpoint
        else:
            lower_offset = midpoint

    event_state = integrate_step(time, state, upper_offset, params)
    event_state[0] = boundary_angle
    return upper_offset, event_state


def simulate_hybrid_dynamics(
    initial_state: np.ndarray,
    timestep: float,
    duration: float,
    params: dict[str, float],
    rest_speed_tolerance: float | None = None,
) -> HybridSimulation:
    """Simulate swings, forward/backward impacts, and Zeno standing.

    Repeated low-speed impacts converge to a two-spoke standing configuration in
    finite physical time. The absorbing ``double support`` mode avoids attempting
    to numerically resolve infinitely many impacts.
    """
    if timestep <= 0.0 or duration <= 0.0:
        raise ValueError("timestep and duration must both be positive")

    speed_scale = np.sqrt(params["gravity"] / params["length"])
    if rest_speed_tolerance is None:
        rest_speed_tolerance = 1e-3 * speed_scale

    step_count = int(np.ceil(duration / timestep))
    time = np.linspace(0.0, duration, step_count + 1)
    state = np.empty((2, step_count + 1), dtype=float)
    state[:, 0] = np.asarray(initial_state, dtype=float)
    lower_angle = wheel.post_impact_angle(params)
    upper_angle = wheel.impact_angle(params)

    impact_times: list[float] = []
    impact_directions: list[int] = []
    pre_impact_states: list[np.ndarray] = []
    post_impact_states: list[np.ndarray] = []
    double_support = False

    def record_impact(
        event_time: float, pre_state: np.ndarray, direction: int
    ) -> np.ndarray:
        if direction > 0:
            post_state = wheel.reset_state(pre_state, params)
        else:
            post_state = wheel.reset_state_backward(pre_state, params)
        impact_times.append(event_time)
        impact_directions.append(direction)
        pre_impact_states.append(pre_state.copy())
        post_impact_states.append(post_state.copy())
        return post_state

    def should_collapse_to_standing(post_state: np.ndarray) -> bool:
        return (
            classify_section_speed(
                post_state[1],
                params,
                tolerance=rest_speed_tolerance,
            )
            == "standing fixed point"
        )

    for index in range(step_count):
        current_time = time[index]
        current_state = state[:, index].copy()
        step_size = time[index + 1] - current_time
        just_impacted = False

        if double_support:
            state[:, index + 1] = current_state
            continue

        if current_state[0] >= upper_angle and current_state[1] > 0.0:
            current_state[0] = upper_angle
            current_state = record_impact(current_time, current_state, 1)
            just_impacted = True
        elif current_state[0] <= lower_angle and current_state[1] < 0.0:
            current_state[0] = lower_angle
            current_state = record_impact(current_time, current_state, -1)
            just_impacted = True

        if just_impacted and should_collapse_to_standing(current_state):
            current_state[1] = 0.0
            state[:, index + 1] = current_state
            double_support = True
            continue

        candidate_state = integrate_step(current_time, current_state, step_size, params)
        crossed_forward = (
            current_state[0] < upper_angle
            and candidate_state[0] >= upper_angle
            and candidate_state[1] > 0.0
        )
        crossed_backward = (
            current_state[0] > lower_angle
            and candidate_state[0] <= lower_angle
            and candidate_state[1] < 0.0
        )

        if crossed_forward or crossed_backward:
            direction = 1 if crossed_forward else -1
            boundary = upper_angle if crossed_forward else lower_angle
            event_offset, pre_state = locate_boundary_crossing(
                current_time,
                current_state,
                step_size,
                boundary,
                direction,
                params,
            )
            post_state = record_impact(
                current_time + event_offset, pre_state, direction
            )
            if should_collapse_to_standing(post_state):
                post_state[1] = 0.0
                candidate_state = post_state
                double_support = True
            else:
                candidate_state = integrate_step(
                    current_time + event_offset,
                    post_state,
                    step_size - event_offset,
                    params,
                )

        state[:, index + 1] = candidate_state

    pre_array = np.asarray(pre_impact_states, dtype=float).reshape(-1, 2)
    post_array = np.asarray(post_impact_states, dtype=float).reshape(-1, 2)
    if double_support:
        outcome = "standing fixed point"
    elif post_array.size:
        outcome = classify_section_speed(post_array[-1, 1], params)
    else:
        outcome = "undetermined"

    return HybridSimulation(
        time=time,
        state=state,
        impact_times=np.asarray(impact_times),
        impact_directions=np.asarray(impact_directions, dtype=int),
        pre_impact_states=pre_array,
        post_impact_states=post_array,
        outcome=outcome,
    )


def simulate_one_downhill_step(
    post_impact_speed: float,
    params: dict[str, float],
    timestep: float = 1e-3,
    maximum_time: float = 12.0,
    record_trajectory: bool = False,
) -> StepSimulation | None:
    """Numerically evaluate one rolling Poincare return.

    ``None`` means that the wheel reverses and contacts the uphill spoke instead.
    """
    lower_angle = wheel.post_impact_angle(params)
    upper_angle = wheel.impact_angle(params)
    state = np.array([lower_angle, post_impact_speed], dtype=float)
    time = 0.0
    times = [time]
    states = [state.copy()]

    while time < maximum_time:
        step_size = min(timestep, maximum_time - time)
        candidate_state = integrate_step(time, state, step_size, params)
        crossed_forward = (
            state[0] < upper_angle
            and candidate_state[0] >= upper_angle
            and candidate_state[1] > 0.0
        )
        if crossed_forward:
            event_offset, pre_state = locate_boundary_crossing(
                time, state, step_size, upper_angle, 1, params
            )
            impact_time = time + event_offset
            post_state = wheel.reset_state(pre_state, params)
            times.append(impact_time)
            states.append(pre_state.copy())
            return StepSimulation(
                time=np.asarray(times),
                state=np.asarray(states).T,
                impact_time=impact_time,
                pre_impact_state=pre_state,
                post_impact_state=post_state,
            )

        crossed_backward = (
            state[0] > lower_angle
            and candidate_state[0] <= lower_angle
            and candidate_state[1] < 0.0
        )
        if crossed_backward:
            return None

        time += step_size
        state = candidate_state
        if record_trajectory:
            times.append(time)
            states.append(state.copy())

    return None


def calculate_numerical_return_map(
    post_impact_speeds: np.ndarray,
    params: dict[str, float],
    timestep: float = 1e-3,
) -> np.ndarray:
    """Sample the positive-speed rolling branch using RK4 and event location."""
    mapped_speeds = np.full_like(post_impact_speeds, np.nan, dtype=float)
    for index, speed in np.ndenumerate(post_impact_speeds):
        step = simulate_one_downhill_step(float(speed), params, timestep=timestep)
        if step is not None:
            mapped_speeds[index] = step.post_impact_state[1]
    return mapped_speeds


def find_numerical_fixed_point(
    params: dict[str, float], timestep: float = 1e-3
) -> float:
    """Find the numerical rolling-map fixed point by bisection."""
    expected = wheel.fixed_point_speed(params)
    if not np.isfinite(expected):
        return float("nan")

    threshold = wheel.minimum_crossing_speed(params)
    lower_speed = max(1.0001 * threshold, 0.7 * expected)
    upper_speed = 1.5 * expected

    def residual(speed: float) -> float:
        mapped = calculate_numerical_return_map(np.asarray([speed]), params, timestep)[
            0
        ]
        return float(mapped - speed)

    lower_residual = residual(lower_speed)
    upper_residual = residual(upper_speed)
    if not np.isfinite(lower_residual) or lower_residual <= 0.0:
        lower_speed = 0.5 * (threshold + expected)
        lower_residual = residual(lower_speed)
    if lower_residual * upper_residual > 0.0:
        raise RuntimeError("could not bracket the return-map fixed point")

    for _ in range(24):
        midpoint = 0.5 * (lower_speed + upper_speed)
        if residual(midpoint) > 0.0:
            lower_speed = midpoint
        else:
            upper_speed = midpoint
    return 0.5 * (lower_speed + upper_speed)


def estimate_numerical_floquet_multiplier(
    fixed_point_speed: float,
    params: dict[str, float],
    relative_perturbation: float = 0.01,
    timestep: float = 1e-3,
) -> tuple[float, float, float]:
    """Return left, right, and centered two-sided slopes near the fixed point."""
    perturbation = relative_perturbation * fixed_point_speed
    sample = fixed_point_speed + perturbation * np.array([-1.0, 0.0, 1.0])
    mapped = calculate_numerical_return_map(sample, params, timestep)
    left_slope = (mapped[1] - mapped[0]) / perturbation
    right_slope = (mapped[2] - mapped[1]) / perturbation
    centered_slope = (mapped[2] - mapped[0]) / (2.0 * perturbation)
    return float(left_slope), float(right_slope), float(centered_slope)


def estimate_event_map_floquet_multiplier(params: dict[str, float]) -> float:
    """Use a centered difference of the event-driven map for fast sweeps."""
    fixed_speed = wheel.fixed_point_speed(params)
    if not np.isfinite(fixed_speed):
        return float("nan")
    perturbation = 0.005 * fixed_speed
    sample = fixed_speed + perturbation * np.array([-1.0, 1.0])
    mapped = wheel.step_to_step_return_map(sample, params)
    return float((mapped[1] - mapped[0]) / (2.0 * perturbation))


def cell_centers(lower: float, upper: float, count: int) -> np.ndarray:
    """Return evenly spaced cell centers, avoiding ambiguous guard endpoints."""
    edges = np.linspace(lower, upper, count + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def classify_section_speed(
    section_speed: float,
    params: dict[str, float],
    maximum_impacts: int = 150,
    tolerance: float | None = None,
) -> str:
    """Classify one signed post-impact speed by iterating the contact map."""
    speed_scale = np.sqrt(params["gravity"] / params["length"])
    if tolerance is None:
        tolerance = 1e-3 * speed_scale
    rolling_speed = wheel.fixed_point_speed(params)
    downhill_threshold = wheel.minimum_crossing_speed(params)
    current_speed = float(section_speed)

    for _ in range(maximum_impacts):
        if abs(current_speed) <= tolerance:
            return "standing fixed point"
        if (
            np.isfinite(rolling_speed)
            and current_speed > downhill_threshold
            and abs(current_speed - rolling_speed) <= tolerance
        ):
            return "rolling limit cycle"
        current_speed = wheel.step_to_step_return_map(current_speed, params)
        if not np.isfinite(current_speed):
            return "undetermined"
    return "undetermined"


def estimate_region_of_attraction(
    params: dict[str, float],
    angle_points: int = 121,
    speed_points: int = 181,
    normalized_speed_limit: float = 1.25,
    maximum_impacts: int = 150,
) -> BasinEstimate:
    """Brute-force an event-driven simulation from every state-grid cell.

    Swing energy gives the next directional collision exactly, then the full
    three-branch contact map is iterated until standing or rolling convergence.
    This is both faster and more accurate than using tiny fixed timesteps around
    the nonsmooth impacts.
    """
    if params["damping_coeff"] != 0.0:
        raise ValueError("the event-driven RoA estimate requires zero damping")

    lower_angle = wheel.post_impact_angle(params)
    upper_angle = wheel.impact_angle(params)
    speed_scale = np.sqrt(params["gravity"] / params["length"])
    angles = cell_centers(lower_angle, upper_angle, angle_points)
    angular_velocities = cell_centers(
        -normalized_speed_limit * speed_scale,
        normalized_speed_limit * speed_scale,
        speed_points,
    )
    angle_grid, speed_grid = np.meshgrid(angles, angular_velocities)

    gravity = params["gravity"]
    length = params["length"]
    energy_above_upright = (
        0.5 * length**2 * speed_grid**2
        + gravity * length * np.cos(angle_grid)
        - gravity * length
    )
    energy_tolerance = 1e-12
    moving_forward = speed_grid > 0.0
    moving_backward = speed_grid < 0.0
    approaches_separatrix = (
        (moving_forward & (angle_grid < 0.0)) | (moving_backward & (angle_grid > 0.0))
    ) & (np.abs(energy_above_upright) <= energy_tolerance)
    approaches_separatrix |= (np.abs(angle_grid) <= energy_tolerance) & (
        speed_grid == 0.0
    )

    reaches_forward_guard = (
        moving_forward
        & ((angle_grid >= 0.0) | (energy_above_upright > energy_tolerance))
    ) | (
        moving_backward
        & (angle_grid > 0.0)
        & (energy_above_upright < -energy_tolerance)
    )
    reaches_forward_guard |= (speed_grid == 0.0) & (angle_grid > 0.0)
    reaches_backward_guard = ~(reaches_forward_guard | approaches_separatrix)

    collision_factor = np.cos(2.0 * params["alpha"])
    forward_speed_squared = speed_grid**2 + 2.0 * gravity / length * (
        np.cos(angle_grid) - np.cos(upper_angle)
    )
    backward_speed_squared = speed_grid**2 + 2.0 * gravity / length * (
        np.cos(angle_grid) - np.cos(lower_angle)
    )
    section_speed = np.zeros_like(speed_grid)
    section_speed[reaches_forward_guard] = collision_factor * np.sqrt(
        np.maximum(forward_speed_squared[reaches_forward_guard], 0.0)
    )
    section_speed[reaches_backward_guard] = -collision_factor * np.sqrt(
        np.maximum(backward_speed_squared[reaches_backward_guard], 0.0)
    )

    attractors = np.full(angle_grid.shape, UNRESOLVED, dtype=np.int8)
    active = ~approaches_separatrix
    rest_tolerance = 1e-3 * speed_scale
    rolling_tolerance = 1e-3 * speed_scale
    rolling_speed = wheel.fixed_point_speed(params)
    rolling_threshold = wheel.minimum_crossing_speed(params)

    for _ in range(maximum_impacts):
        standing = active & (np.abs(section_speed) <= rest_tolerance)
        attractors[standing] = STANDING
        active[standing] = False

        if np.isfinite(rolling_speed):
            rolling = active & (
                (section_speed > rolling_threshold)
                & (np.abs(section_speed - rolling_speed) <= rolling_tolerance)
            )
            attractors[rolling] = ROLLING
            active[rolling] = False

        if not np.any(active):
            break
        active_indices = np.flatnonzero(active)
        next_speed = wheel.step_to_step_return_map(
            section_speed.flat[active_indices], params
        )
        undefined = ~np.isfinite(next_speed)
        active.flat[active_indices[undefined]] = False
        section_speed.flat[active_indices[~undefined]] = next_speed[~undefined]

    return BasinEstimate(angles, angular_velocities, attractors)


def calculate_critical_slope(number_of_spokes: int) -> float:
    """Analytic onset slope for a finite-period passive rolling gait."""
    alpha = np.pi / number_of_spokes
    return float(2.0 * np.arctan(np.tan(alpha / 2.0) * np.tan(alpha) ** 2))


def draw_model_sketch(params: dict[str, float], output_path: Path) -> None:
    """Draw the annotated geometry at a downhill collision."""
    alpha = params["alpha"]
    slope = params["gamma"]
    length = params["length"]
    pre_angle = slope + alpha
    post_angle = slope - alpha
    old_contact = np.array([0.0, 0.0])
    hub = length * np.array([np.sin(pre_angle), np.cos(pre_angle)])
    new_contact = hub - length * np.array([np.sin(post_angle), np.cos(post_angle)])

    figure, axis = plt.subplots(figsize=(7.4, 4.5))
    ground_x = np.linspace(-0.8, 2.2, 100)
    axis.plot(ground_x, -np.tan(slope) * ground_x, color="#6b5d4d", lw=3)
    axis.plot(
        [old_contact[0], hub[0]],
        [old_contact[1], hub[1]],
        color="#1f77b4",
        lw=4,
        label="old stance spoke",
    )
    axis.plot(
        [new_contact[0], hub[0]],
        [new_contact[1], hub[1]],
        color="#f28e2b",
        lw=4,
        label="new stance spoke",
    )
    axis.scatter(*old_contact, color="#1f77b4", s=55, zorder=4)
    axis.scatter(*new_contact, color="#f28e2b", s=55, zorder=4)
    axis.scatter(*hub, color="black", s=180, zorder=5)
    axis.text(hub[0] + 0.04, hub[1] + 0.04, "point mass $m$", fontsize=11)
    axis.plot(
        [old_contact[0], old_contact[0]],
        [old_contact[1], old_contact[1] + 1.15],
        ls="--",
        color="0.55",
    )
    axis.add_patch(
        Arc(
            old_contact,
            0.7,
            0.7,
            theta1=90.0 - np.degrees(pre_angle),
            theta2=90.0,
            color="#1f77b4",
            lw=1.8,
        )
    )
    axis.text(0.07, 0.42, r"$\theta^- = \gamma+\alpha$", color="#1f77b4")
    axis.add_patch(
        Arc(
            old_contact,
            1.0,
            0.55,
            theta1=-np.degrees(slope),
            theta2=0.0,
            color="#6b5d4d",
            lw=1.8,
        )
    )
    axis.text(0.43, -0.02, r"$\gamma$", color="#6b5d4d")
    axis.annotate(
        r"$g$",
        xy=(hub[0], hub[1] - 0.5),
        xytext=(hub[0], hub[1] - 0.08),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.8},
        ha="center",
        fontsize=12,
    )
    axis.text(0.5 * hub[0] - 0.1, 0.5 * hub[1], r"$l$", fontsize=12)
    axis.text(
        hub[0] + 0.18,
        hub[1] - 0.27,
        r"angle between spokes $=2\alpha=2\pi/N$",
        fontsize=10,
    )
    axis.text(
        0.03,
        0.95,
        r"state $x=[\theta,\dot\theta]$; downhill is positive",
        transform=axis.transAxes,
        fontsize=11,
        va="top",
    )
    axis.set_aspect("equal")
    axis.set(xlim=(-0.55, 2.0), ylim=(-0.45, 1.35))
    axis.axis("off")
    axis.legend(loc="lower left", frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_sanity_checks(
    simulation: HybridSimulation,
    params: dict[str, float],
    output_path: Path,
) -> None:
    """Plot state, phase, energy, and gait-convergence checks."""
    kinetic_energy, potential_energy = wheel.calculate_energy(simulation.state, params)
    total_energy = kinetic_energy + potential_energy
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.2))

    axes[0, 0].plot(simulation.time, simulation.state[0], label=r"$\theta$")
    axes[0, 0].plot(simulation.time, simulation.state[1], label=r"$\dot\theta$")
    for impact_time in simulation.impact_times:
        axes[0, 0].axvline(impact_time, color="0.82", lw=0.6)
    axes[0, 0].set(xlabel="time (s)", ylabel="state", title="Hybrid trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(simulation.state[0], simulation.state[1], lw=1.2)
    axes[0, 1].set(
        xlabel=r"angle $\theta$ (rad)",
        ylabel=r"angular velocity $\dot\theta$ (rad/s)",
        title="Phase portrait",
    )

    axes[1, 0].plot(simulation.time, kinetic_energy, label="kinetic")
    axes[1, 0].plot(simulation.time, potential_energy, label="potential")
    axes[1, 0].plot(simulation.time, total_energy, label="total", lw=2)
    axes[1, 0].set(
        xlabel="time (s)", ylabel="energy (J)", title="Global mechanical energy"
    )
    axes[1, 0].legend()

    post_speeds = simulation.post_impact_states[:, 1]
    axes[1, 1].plot(
        np.arange(1, post_speeds.size + 1),
        post_speeds,
        "o-",
        label="simulated",
    )
    fixed_speed = wheel.fixed_point_speed(params)
    axes[1, 1].axhline(fixed_speed, color="black", ls="--", label="fixed point")
    axes[1, 1].set(
        xlabel="impact number",
        ylabel="post-impact speed (rad/s)",
        title="Convergence to the rolling gait",
    )
    axes[1, 1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_region_of_attraction(
    basin: BasinEstimate,
    params: dict[str, float],
    numerical_fixed_point: float,
    output_path: Path,
) -> None:
    """Plot the two basins and representative attractors in physical state space."""
    figure, axis = plt.subplots(figsize=(8.6, 5.6))
    angle_grid, speed_grid = np.meshgrid(basin.angles, basin.angular_velocities)
    color_map = ListedColormap(["#4c78a8", "#f58518", "#d9d9d9"])
    axis.pcolormesh(
        angle_grid,
        speed_grid,
        basin.attractors,
        cmap=color_map,
        shading="nearest",
        vmin=-0.5,
        vmax=2.5,
    )

    rolling_step = simulate_one_downhill_step(
        numerical_fixed_point,
        params,
        timestep=5e-4,
        record_trajectory=True,
    )
    if rolling_step is not None:
        axis.plot(rolling_step.state[0], rolling_step.state[1], color="white", lw=4)
        axis.plot(
            rolling_step.state[0],
            rolling_step.state[1],
            color="black",
            lw=1.5,
            label="rolling limit cycle",
        )
        axis.plot(
            [rolling_step.pre_impact_state[0], rolling_step.post_impact_state[0]],
            [rolling_step.pre_impact_state[1], rolling_step.post_impact_state[1]],
            color="black",
            ls=":",
            lw=1.5,
        )

    axis.scatter(
        [wheel.post_impact_angle(params), wheel.impact_angle(params)],
        [0.0, 0.0],
        marker="s",
        s=75,
        facecolor="white",
        edgecolor="black",
        lw=1.5,
        label="standing fixed point (double support)",
        zorder=5,
    )
    axis.scatter(
        0.0,
        0.0,
        marker="x",
        s=70,
        color="black",
        label="unstable single-spoke equilibrium",
        zorder=5,
    )
    patches = [
        Patch(color="#4c78a8", label="converges to standing"),
        Patch(color="#f58518", label="converges to rolling"),
        Patch(color="#d9d9d9", label="separatrix / unresolved"),
    ]
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(patches + handles, [patch.get_label() for patch in patches] + labels)
    axis.set(
        xlim=(wheel.post_impact_angle(params), wheel.impact_angle(params)),
        xlabel=r"initial angle $\theta_0$ (rad)",
        ylabel=r"initial angular velocity $\dot\theta_0$ (rad/s)",
        title=(
            rf"RoA: $N={params['N']}$, $\gamma={params['gamma']:.2f}$ rad; "
            f"rolling fraction = {basin.rolling_fraction:.3f}"
        ),
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_return_map(
    input_speeds: np.ndarray,
    mapped_speeds: np.ndarray,
    params: dict[str, float],
    fixed_speed: float,
    output_path: Path,
) -> None:
    """Plot numerical and exact return maps, identity, and both fixed points."""
    figure, axis = plt.subplots(figsize=(7.3, 6.0))
    uphill_threshold, downhill_threshold = wheel.return_map_thresholds(params)
    branch_bounds = [
        (input_speeds.min(), uphill_threshold),
        (uphill_threshold, downhill_threshold),
        (downhill_threshold, input_speeds.max()),
    ]
    for branch_index, (lower_bound, upper_bound) in enumerate(branch_bounds):
        branch_input = np.linspace(lower_bound, upper_bound, 170)
        branch_input = branch_input[1:-1]
        branch_output = wheel.step_to_step_return_map(branch_input, params)
        axis.plot(
            branch_input,
            branch_output,
            color="#e45756",
            lw=2,
            label="event map" if branch_index == 0 else None,
        )
    axis.plot(
        input_speeds,
        mapped_speeds,
        "o",
        color="#4c78a8",
        ms=3.5,
        label="RK4 event samples (rolling branch)",
    )
    limits = [float(input_speeds.min()), float(input_speeds.max())]
    axis.plot(limits, limits, "k--", label="identity")
    axis.scatter(
        [0.0, fixed_speed],
        [0.0, fixed_speed],
        s=80,
        color=["#4c78a8", "#f58518"],
        edgecolor="black",
        zorder=5,
        label="fixed points",
    )
    axis.axvline(
        wheel.minimum_crossing_speed(params),
        color="0.45",
        ls=":",
        label="forward-step threshold",
    )
    axis.set(
        xlim=limits,
        ylim=limits,
        xlabel=r"current post-impact speed $\dot\theta_k^+$ (rad/s)",
        ylabel=r"next post-impact speed $\dot\theta_{k+1}^+$ (rad/s)",
        title=rf"Contact return map; rolling fixed point $={fixed_speed:.4f}$ rad/s",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_parameter_sweeps(
    slope_rows: list[dict[str, float]],
    spoke_rows: list[dict[str, float]],
    critical_slope: float,
    output_path: Path,
) -> None:
    """Plot RoA fraction and Floquet multiplier for both required sweeps."""
    slopes = np.asarray([row["slope_rad"] for row in slope_rows])
    slope_roa = np.asarray([row["rolling_fraction"] for row in slope_rows])
    slope_floquet = np.asarray([row["floquet_multiplier"] for row in slope_rows])
    spokes = np.asarray([row["number_of_spokes"] for row in spoke_rows])
    spoke_roa = np.asarray([row["rolling_fraction"] for row in spoke_rows])
    spoke_floquet = np.asarray([row["floquet_multiplier"] for row in spoke_rows])

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex="col")
    axes[0, 0].plot(slopes, slope_roa, "o-", color="#f58518")
    axes[0, 0].axvline(
        critical_slope,
        color="black",
        ls=":",
        label=rf"$\gamma_{{crit}}={critical_slope:.3f}$",
    )
    axes[0, 0].set(ylabel="rolling RoA fraction", title=r"Slope sweep ($N=8$)")
    axes[0, 0].legend()
    axes[1, 0].plot(slopes, slope_floquet, "o-", color="#4c78a8")
    axes[1, 0].axvline(critical_slope, color="black", ls=":")
    axes[1, 0].set(
        xlabel=r"slope $\gamma$ (rad)", ylabel="Floquet multiplier", ylim=(0, 0.85)
    )

    axes[0, 1].plot(spokes, spoke_roa, "o-", color="#f58518")
    axes[0, 1].set(title=r"Spoke sweep ($\gamma=0.20$ rad)")
    axes[1, 1].plot(spokes, spoke_floquet, "o-", color="#4c78a8")
    axes[1, 1].set(xlabel="number of spokes $N$", ylim=(0, 0.85))
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.supylabel("metric value")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def calculate_sanity_metrics(
    simulation: HybridSimulation, params: dict[str, float]
) -> dict[str, float]:
    """Quantify energy conservation and collision-reset accuracy."""
    forward_impacts = simulation.impact_directions > 0
    pre_states = simulation.pre_impact_states[forward_impacts]
    post_states = simulation.post_impact_states[forward_impacts]
    if pre_states.size == 0:
        raise RuntimeError("sanity trajectory did not produce a forward impact")

    mass = params["mass"]
    gravity = params["gravity"]
    length = params["length"]
    start_speeds = np.concatenate(([simulation.state[1, 0]], post_states[:-1, 1]))
    start_energy = 0.5 * mass * (
        length * start_speeds
    ) ** 2 + mass * gravity * length * np.cos(wheel.post_impact_angle(params))
    pre_energy = 0.5 * mass * (
        length * pre_states[:, 1]
    ) ** 2 + mass * gravity * length * np.cos(pre_states[:, 0])
    expected_post_speed = pre_states[:, 1] * np.cos(2.0 * params["alpha"])
    return {
        "maximum_swing_energy_drift_joule": float(
            np.max(np.abs(pre_energy - start_energy))
        ),
        "maximum_impact_angle_error_rad": float(
            np.max(np.abs(pre_states[:, 0] - wheel.impact_angle(params)))
        ),
        "maximum_reset_angle_error_rad": float(
            np.max(np.abs(post_states[:, 0] - wheel.post_impact_angle(params)))
        ),
        "maximum_reset_speed_error_rad_per_second": float(
            np.max(np.abs(post_states[:, 1] - expected_post_speed))
        ),
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """Write a deterministic results table."""
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(output_directory: Path, quick: bool = False) -> dict[str, object]:
    """Generate every code, plot, and numeric artifact required by Assignment 1."""
    output_directory.mkdir(parents=True, exist_ok=True)
    params = wheel.generate_params(number_of_spokes=8, slope=0.1)
    integration_step = 2e-3 if quick else 1e-3

    draw_model_sketch(params, output_directory / "model_sketch.png")
    simulation = simulate_hybrid_dynamics(
        np.array([wheel.post_impact_angle(params), 1.8]),
        timestep=integration_step,
        duration=10.0,
        params=params,
    )
    plot_sanity_checks(simulation, params, output_directory / "sanity_checks.png")
    sanity_metrics = calculate_sanity_metrics(simulation, params)

    angle_points = 61 if quick else 121
    speed_points = 81 if quick else 181
    basin = estimate_region_of_attraction(
        params, angle_points=angle_points, speed_points=speed_points
    )
    numerical_fixed_speed = find_numerical_fixed_point(params, integration_step)
    plot_region_of_attraction(
        basin,
        params,
        numerical_fixed_speed,
        output_directory / "region_of_attraction.png",
    )

    map_points = 40 if quick else 75
    map_inputs = np.linspace(-2.2, 2.8, map_points)
    positive_inputs = map_inputs[map_inputs > wheel.minimum_crossing_speed(params)]
    positive_outputs = calculate_numerical_return_map(
        positive_inputs, params, integration_step
    )
    numerical_outputs = np.full_like(map_inputs, np.nan)
    numerical_outputs[map_inputs > wheel.minimum_crossing_speed(params)] = (
        positive_outputs
    )
    plot_return_map(
        map_inputs,
        numerical_outputs,
        params,
        numerical_fixed_speed,
        output_directory / "return_map.png",
    )
    left_slope, right_slope, numerical_multiplier = (
        estimate_numerical_floquet_multiplier(
            numerical_fixed_speed,
            params,
            relative_perturbation=0.01,
            timestep=integration_step,
        )
    )

    sweep_angle_points = 51 if quick else 101
    sweep_speed_points = 81 if quick else 161
    slope_values = np.arange(0.04, 0.361, 0.08 if quick else 0.04)
    slope_rows: list[dict[str, float]] = []
    for slope in slope_values:
        sweep_params = wheel.generate_params(number_of_spokes=8, slope=float(slope))
        sweep_basin = estimate_region_of_attraction(
            sweep_params,
            angle_points=sweep_angle_points,
            speed_points=sweep_speed_points,
        )
        slope_rows.append(
            {
                "slope_rad": float(slope),
                "slope_deg": float(np.degrees(slope)),
                "rolling_fraction": sweep_basin.rolling_fraction,
                "unresolved_fraction": sweep_basin.unresolved_fraction,
                "fixed_point_speed_rad_per_s": float(
                    wheel.fixed_point_speed(sweep_params)
                ),
                "minimum_crossing_speed_rad_per_s": float(
                    wheel.minimum_crossing_speed(sweep_params)
                ),
                "floquet_multiplier": estimate_event_map_floquet_multiplier(
                    sweep_params
                ),
            }
        )

    spoke_rows: list[dict[str, float]] = []
    for number_of_spokes in range(6, 13):
        sweep_params = wheel.generate_params(
            number_of_spokes=number_of_spokes, slope=0.2
        )
        sweep_basin = estimate_region_of_attraction(
            sweep_params,
            angle_points=sweep_angle_points,
            speed_points=sweep_speed_points,
        )
        spoke_rows.append(
            {
                "number_of_spokes": number_of_spokes,
                "alpha_rad": float(sweep_params["alpha"]),
                "critical_slope_rad": calculate_critical_slope(number_of_spokes),
                "rolling_fraction": sweep_basin.rolling_fraction,
                "unresolved_fraction": sweep_basin.unresolved_fraction,
                "fixed_point_speed_rad_per_s": float(
                    wheel.fixed_point_speed(sweep_params)
                ),
                "minimum_crossing_speed_rad_per_s": float(
                    wheel.minimum_crossing_speed(sweep_params)
                ),
                "floquet_multiplier": estimate_event_map_floquet_multiplier(
                    sweep_params
                ),
            }
        )

    critical_slope = calculate_critical_slope(8)
    plot_parameter_sweeps(
        slope_rows,
        spoke_rows,
        critical_slope,
        output_directory / "parameter_sweeps.png",
    )
    write_csv(output_directory / "slope_sweep.csv", slope_rows)
    write_csv(output_directory / "spoke_sweep.csv", spoke_rows)

    coarse_basin = estimate_region_of_attraction(
        params, angle_points=61, speed_points=91
    )
    summary: dict[str, object] = {
        "default_parameters": {
            key: float(value) if isinstance(value, (float, np.floating)) else value
            for key, value in params.items()
        },
        "sanity_checks": sanity_metrics,
        "default_region_of_attraction": {
            "angle_points": angle_points,
            "speed_points": speed_points,
            "normalized_angle_limits": [-1.0, 1.0],
            "normalized_speed_limits": [-1.25, 1.25],
            "rolling_fraction": basin.rolling_fraction,
            "unresolved_fraction": basin.unresolved_fraction,
            "coarse_grid_rolling_fraction": coarse_basin.rolling_fraction,
            "grid_refinement_change": abs(
                basin.rolling_fraction - coarse_basin.rolling_fraction
            ),
        },
        "return_map": {
            "minimum_crossing_speed_rad_per_s": float(
                wheel.minimum_crossing_speed(params)
            ),
            "numerical_fixed_point_speed_rad_per_s": numerical_fixed_speed,
            "analytic_fixed_point_speed_rad_per_s": float(
                wheel.fixed_point_speed(params)
            ),
            "left_finite_difference_slope": left_slope,
            "right_finite_difference_slope": right_slope,
            "centered_floquet_multiplier": numerical_multiplier,
            "analytic_floquet_multiplier": float(wheel.floquet_multiplier(params)),
        },
        "sweeps": {
            "critical_slope_for_eight_spokes_rad": critical_slope,
            "slope_grid_count": len(slope_rows),
            "spoke_counts": list(range(6, 13)),
            "spoke_sweep_slope_rad": 0.2,
            "normalized_state_grid": {
                "angle": "(theta - gamma) / alpha in [-1, 1]",
                "angular_velocity": ("omega * sqrt(length / gravity) in [-1.25, 1.25]"),
            },
        },
    }
    with (output_directory / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, allow_nan=True)
        output_file.write("\n")
    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assignment_1_results"),
        help="directory for generated figures and tables",
    )
    parser.add_argument(
        "--quick", action="store_true", help="use smaller grids for a smoke test"
    )
    return parser.parse_args()


def main() -> None:
    """Run the full analysis and print its headline results."""
    arguments = parse_arguments()
    summary = run_analysis(arguments.output_dir, arguments.quick)
    return_map = summary["return_map"]
    basin = summary["default_region_of_attraction"]
    print(f"Wrote Assignment 1 results to {arguments.output_dir.resolve()}")
    print(f"Default rolling RoA fraction: {basin['rolling_fraction']:.4f}")
    print(
        "Return-map fixed point: "
        f"{return_map['numerical_fixed_point_speed_rad_per_s']:.6f} rad/s"
    )
    print(
        f"Centered Floquet multiplier: {return_map['centered_floquet_multiplier']:.6f}"
    )


if __name__ == "__main__":
    main()
