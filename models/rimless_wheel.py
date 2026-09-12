"""Dynamics and analytical tools for a passive rimless wheel."""

from collections.abc import Mapping

import numpy as np


def _positive_number(value, name):
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _nonnegative_number(value, name):
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return number


def _validated_state(state):
    state_array = np.asarray(state, dtype=float)
    if state_array.ndim == 0 or state_array.shape[0] != 2:
        raise ValueError("state must have shape (2,) or (2, ...)")
    if not np.all(np.isfinite(state_array)):
        raise ValueError("state must contain only finite values")
    return state_array


def _physical_parameters(params):
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")

    gravity = _positive_number(params["gravity"], "gravity")
    length = _positive_number(params["length"], "length")
    mass = _positive_number(params["mass"], "mass")
    damping = _nonnegative_number(params["damping_coeff"], "damping_coeff")
    return gravity, length, mass, damping


def _geometry_parameters(params):
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping")

    number_of_spokes = params["N"]
    if isinstance(number_of_spokes, (bool, np.bool_)):
        raise TypeError("N must be an integer greater than or equal to 3")
    try:
        integer_spokes = int(number_of_spokes)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("N must be an integer greater than or equal to 3") from error
    if integer_spokes != number_of_spokes or integer_spokes < 3:
        raise ValueError("N must be an integer greater than or equal to 3")

    slope = float(params["gamma"])
    if not np.isfinite(slope):
        raise ValueError("gamma must be finite")

    expected_alpha = np.pi / integer_spokes
    alpha = float(params.get("alpha", expected_alpha))
    if not np.isfinite(alpha) or not np.isclose(
        alpha, expected_alpha, rtol=1e-12, atol=1e-15
    ):
        raise ValueError("alpha must equal pi / N")
    return integer_spokes, slope, expected_alpha


def _require_undamped(params):
    gravity, length, _, damping = _physical_parameters(params)
    if damping != 0.0:
        raise ValueError(
            "analytical gait helpers require the undamped model (damping_coeff == 0)"
        )
    return gravity, length


def generate_params(
    number_of_spokes=8,
    slope=0.1,
    gravity=9.81,
    length=1.0,
    mass=1.0,
    damping=0.0,
):
    """Return a validated parameter dictionary for the rimless wheel.

    The short keys ``N``, ``gamma``, and ``alpha`` are retained to remain
    compatible with the original assignment code.
    """
    if isinstance(number_of_spokes, (bool, np.bool_)):
        raise TypeError("number_of_spokes must be an integer >= 3")
    try:
        integer_spokes = int(number_of_spokes)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("number_of_spokes must be an integer >= 3") from error
    if integer_spokes != number_of_spokes or integer_spokes < 3:
        raise ValueError("number_of_spokes must be an integer >= 3")

    slope = float(slope)
    if not np.isfinite(slope):
        raise ValueError("slope must be finite")

    params = {
        "N": integer_spokes,
        "gamma": slope,
        "gravity": _positive_number(gravity, "gravity"),
        "length": _positive_number(length, "length"),
        "mass": _positive_number(mass, "mass"),
        "damping_coeff": _nonnegative_number(damping, "damping"),
        "alpha": np.pi / integer_spokes,
    }
    return params


def impact_angle(params):
    """Return the downhill (upper-angle) contact guard ``gamma + alpha``."""
    _, slope, alpha = _geometry_parameters(params)
    return float(slope + alpha)


def post_impact_angle(params):
    """Return the angle after a downhill impact, ``gamma - alpha``."""
    _, slope, alpha = _geometry_parameters(params)
    return float(slope - alpha)


def uphill_impact_angle(params):
    """Return the uphill (lower-angle) contact guard ``gamma - alpha``."""
    return post_impact_angle(params)


def dynamics(time, state, params):
    """Evaluate the continuous inverted-pendulum dynamics.

    ``state`` may have shape ``(2,)`` or ``(2, ...)``. The first row is the
    stance-spoke angle and the second row is angular velocity.
    """
    del time  # The dynamics are autonomous, but integrators pass time in.
    state_array = _validated_state(state)
    gravity, length, mass, damping = _physical_parameters(params)

    angle = state_array[0]
    angular_velocity = state_array[1]
    angular_acceleration = (
        gravity / length * np.sin(angle)
        - damping / (mass * length**2) * angular_velocity
    )
    return np.stack((angular_velocity, angular_acceleration), axis=0)


def need_reset(state, params):
    """Return whether each state has reached the forward impact guard.

    Contact occurs at ``theta = gamma + alpha`` only while the wheel is
    rotating downhill (positive angular velocity). For a batched state, the
    returned Boolean array has the batch shape.
    """
    state_array = _validated_state(state)
    impact = (state_array[0] >= impact_angle(params)) & (state_array[1] > 0.0)
    return bool(impact) if np.ndim(impact) == 0 else impact


def need_reset_backward(state, params):
    """Return whether each state has reached the uphill impact guard."""
    state_array = _validated_state(state)
    impact = (state_array[0] <= uphill_impact_angle(params)) & (state_array[1] < 0.0)
    return bool(impact) if np.ndim(impact) == 0 else impact


def need_backward_reset(state, params):
    """Alias for :func:`need_reset_backward`."""
    return need_reset_backward(state, params)


def reset_state(state, params):
    """Apply the perfectly plastic spoke-switch collision map.

    The post-impact angle is set exactly to ``gamma - alpha`` so a numerical
    integration step that overshoots the guard does not preserve angle error.
    Angular momentum about the new contact point gives
    ``omega_plus = cos(2 * alpha) * omega_minus``.
    """
    state_array = _validated_state(state)
    _, slope, alpha = _geometry_parameters(params)
    post_impact_angle = np.full_like(state_array[1], slope - alpha)
    post_impact_velocity = np.cos(2.0 * alpha) * state_array[1]
    return np.stack((post_impact_angle, post_impact_velocity), axis=0)


def reset_state_backward(state, params):
    """Apply the spoke-switch map at an uphill/backward collision.

    The new stance angle is the upper angle ``gamma + alpha`` and angular
    momentum about the new contact point gives the same ``cos(2 * alpha)``
    velocity factor as a downhill collision.
    """
    state_array = _validated_state(state)
    _, _, alpha = _geometry_parameters(params)
    post_impact_angle = np.full_like(state_array[1], impact_angle(params))
    post_impact_velocity = np.cos(2.0 * alpha) * state_array[1]
    return np.stack((post_impact_angle, post_impact_velocity), axis=0)


def reset_backward_state(state, params):
    """Alias for :func:`reset_state_backward`."""
    return reset_state_backward(state, params)


def calculate_energy(state, params):
    """Return kinetic and potential energy for a state or trajectory.

    For an array with shape ``(2, ..., samples)``, the last axis is interpreted
    as time. Forward and backward spoke resets are detected from the angle
    discontinuity so potential energy includes the contact point's vertical
    displacement along the slope. A single state uses the current contact point
    as zero height.
    """
    state_array = _validated_state(state)
    gravity, length, mass, _ = _physical_parameters(params)
    _, slope, alpha = _geometry_parameters(params)

    angle = state_array[0]
    angular_velocity = state_array[1]
    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2

    if angle.ndim == 0:
        contact_height = 0.0
    else:
        angle_jumps = np.diff(angle, axis=-1)
        velocity_before = angular_velocity[..., :-1]
        velocity_after = angular_velocity[..., 1:]
        downhill_resets = (
            (angle_jumps < -alpha) & (velocity_before > 0.0) & (velocity_after > 0.0)
        )
        uphill_resets = (
            (angle_jumps > alpha) & (velocity_before < 0.0) & (velocity_after < 0.0)
        )
        step_changes = downhill_resets.astype(int) - uphill_resets.astype(int)
        initial_step = np.zeros(angle.shape[:-1] + (1,), dtype=int)
        step_number = np.concatenate(
            (initial_step, np.cumsum(step_changes, axis=-1)), axis=-1
        )
        contact_height_drop = 2.0 * length * np.sin(alpha) * np.sin(slope)
        contact_height = -step_number * contact_height_drop

    hub_height = contact_height + length * np.cos(angle)
    potential_energy = mass * gravity * hub_height
    return kinetic_energy, potential_energy


def minimum_crossing_speed(params):
    """Return the least forward post-impact speed that reaches the next step.

    This is the conservative energy threshold between the post-impact angle
    ``gamma - alpha`` and the next guard ``gamma + alpha``. At the threshold,
    a trajectory that encounters an upright configuration lies on the
    separatrix, so finite-time crossing requires a speed infinitesimally above
    the returned value.
    """
    gravity, length = _require_undamped(params)
    _, slope, alpha = _geometry_parameters(params)
    start_angle = slope - alpha
    guard_angle = slope + alpha

    first_cosine_peak = np.ceil(start_angle / (2.0 * np.pi))
    last_cosine_peak = np.floor(guard_angle / (2.0 * np.pi))
    if first_cosine_peak <= last_cosine_peak:
        maximum_cosine = 1.0
    else:
        maximum_cosine = max(np.cos(start_angle), np.cos(guard_angle))

    required_squared_speed = (
        2.0 * gravity / length * (maximum_cosine - np.cos(start_angle))
    )
    return float(np.sqrt(max(0.0, required_squared_speed)))


def minimum_uphill_crossing_speed(params):
    """Return the speed magnitude needed to cross upright while moving uphill.

    The wheel starts just after a backward impact at ``gamma + alpha`` and has
    negative angular velocity. The returned value is a positive magnitude.
    """
    gravity, length = _require_undamped(params)
    _, slope, alpha = _geometry_parameters(params)
    if not 0.0 <= slope < alpha:
        raise ValueError("the signed return map requires 0 <= gamma < alpha")
    required_squared_speed = 2.0 * gravity / length * (1.0 - np.cos(slope + alpha))
    return float(np.sqrt(max(0.0, required_squared_speed)))


def return_map_thresholds(params):
    """Return the uphill and downhill separatrix speeds ``(w2, w1)``."""
    _, slope, alpha = _geometry_parameters(params)
    if not 0.0 <= slope < alpha:
        raise ValueError("the signed return map requires 0 <= gamma < alpha")
    uphill_threshold = -minimum_uphill_crossing_speed(params)
    downhill_threshold = minimum_crossing_speed(params)
    return uphill_threshold, downhill_threshold


def step_to_step_return_map(post_impact_speed, params):
    """Map one signed post-impact angular velocity to the next.

    For ``0 <= gamma < alpha``, the map contains downhill rolling, a standing
    branch that reverses before upright, and uphill rolling. Scalar and
    NumPy-array inputs are accepted. The two exact separatrix velocities map to
    ``nan`` because they approach the upright equilibrium asymptotically.
    """
    gravity, length = _require_undamped(params)
    _, slope, alpha = _geometry_parameters(params)
    if not 0.0 <= slope < alpha:
        raise ValueError("the signed return map requires 0 <= gamma < alpha")
    speeds = np.asarray(post_impact_speed, dtype=float)
    if not np.all(np.isfinite(speeds)):
        raise ValueError("post_impact_speed must contain only finite values")

    collision_factor = np.cos(2.0 * alpha)
    squared_speed_gain = 4.0 * gravity / length * np.sin(alpha) * np.sin(slope)
    if collision_factor <= 0.0:
        raise ValueError("the signed return map requires cos(2 * alpha) > 0")
    uphill_threshold, downhill_threshold = return_map_thresholds(params)

    result = np.full_like(speeds, np.nan, dtype=float)
    downhill = speeds > downhill_threshold
    standing = (speeds > uphill_threshold) & (speeds < downhill_threshold)
    uphill = speeds < uphill_threshold
    with np.errstate(invalid="ignore"):
        result[downhill] = collision_factor * np.sqrt(
            speeds[downhill] ** 2 + squared_speed_gain
        )
        result[standing] = -collision_factor * speeds[standing]
        result[uphill] = -collision_factor * np.sqrt(
            speeds[uphill] ** 2 - squared_speed_gain
        )
    return float(result) if result.ndim == 0 else result


def return_map(post_impact_speed, params):
    """Alias for :func:`step_to_step_return_map`."""
    return step_to_step_return_map(post_impact_speed, params)


def fixed_point_speed(params):
    """Return the post-impact speed of the passive rolling limit cycle.

    ``nan`` is returned when the algebraic fixed point is not a reachable
    forward gait (for example, on a slope that is too shallow).
    """
    gravity, length = _require_undamped(params)
    _, slope, alpha = _geometry_parameters(params)
    collision_factor = np.cos(2.0 * alpha)
    squared_speed_gain = 4.0 * gravity / length * np.sin(alpha) * np.sin(slope)
    denominator = 1.0 - collision_factor**2

    if collision_factor <= 0.0 or denominator <= 0.0 or squared_speed_gain <= 0.0:
        return float("nan")

    squared_fixed_speed = collision_factor**2 * squared_speed_gain / denominator
    candidate = float(np.sqrt(squared_fixed_speed))
    threshold = minimum_crossing_speed(params)
    threshold_tolerance = 1e-12 * np.sqrt(gravity / length)
    if candidate <= threshold or np.isclose(
        candidate, threshold, rtol=1e-12, atol=threshold_tolerance
    ):
        return float("nan")
    return candidate


def fixed_speed(params):
    """Alias for :func:`fixed_point_speed`."""
    return fixed_point_speed(params)


def floquet_multiplier(params):
    """Return the local slope of the step-to-step map at its fixed point.

    For the passive undamped wheel this equals ``cos(2 * alpha) ** 2`` and is
    independent of slope. ``nan`` is returned when no reachable gait exists.
    """
    fixed_point = fixed_point_speed(params)
    if np.isnan(fixed_point):
        return float("nan")
    _, _, alpha = _geometry_parameters(params)
    return float(np.cos(2.0 * alpha) ** 2)
