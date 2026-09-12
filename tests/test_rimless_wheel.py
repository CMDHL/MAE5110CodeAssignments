import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import rimless_wheel


def test_generate_default_parameters_and_geometry():
    params = rimless_wheel.generate_params()

    assert params["N"] == 8
    assert params["gamma"] == pytest.approx(0.1)
    assert params["alpha"] == pytest.approx(np.pi / 8.0)
    assert params["gravity"] == pytest.approx(9.81)
    assert params["length"] == pytest.approx(1.0)
    assert params["mass"] == pytest.approx(1.0)
    assert params["damping_coeff"] == pytest.approx(0.0)


def test_dynamics_supports_single_and_batched_states():
    params = rimless_wheel.generate_params(damping=0.2, mass=2.0, length=1.5)
    states = np.array([[0.0, 0.2, -0.3], [1.0, -0.5, 0.25]])

    derivatives = rimless_wheel.dynamics(0.0, states, params)
    expected_acceleration = (
        params["gravity"] / params["length"] * np.sin(states[0])
        - params["damping_coeff"] / (params["mass"] * params["length"] ** 2) * states[1]
    )

    assert derivatives.shape == states.shape
    np.testing.assert_allclose(derivatives[0], states[1])
    np.testing.assert_allclose(derivatives[1], expected_acceleration)
    np.testing.assert_allclose(
        rimless_wheel.dynamics(0.0, states[:, 1], params), derivatives[:, 1]
    )


def test_impact_guards_are_directional_and_vectorized():
    params = rimless_wheel.generate_params()
    upper_angle = rimless_wheel.impact_angle(params)
    lower_angle = rimless_wheel.uphill_impact_angle(params)

    assert not rimless_wheel.need_reset(np.array([upper_angle - 1e-8, 1.0]), params)
    assert rimless_wheel.need_reset(np.array([upper_angle, 1.0]), params)
    assert not rimless_wheel.need_reset(np.array([upper_angle, 0.0]), params)
    assert not rimless_wheel.need_reset(np.array([upper_angle, -1.0]), params)
    assert not rimless_wheel.need_reset_backward(
        np.array([lower_angle + 1e-8, -1.0]), params
    )
    assert rimless_wheel.need_reset_backward(np.array([lower_angle, -1.0]), params)
    assert not rimless_wheel.need_reset_backward(np.array([lower_angle, 1.0]), params)
    assert not rimless_wheel.need_backward_reset(np.array([lower_angle, 0.0]), params)

    states = np.array(
        [
            [upper_angle - 1e-8, upper_angle, upper_angle + 1e-8],
            [1.0, 1.0, -1.0],
        ]
    )
    np.testing.assert_array_equal(
        rimless_wheel.need_reset(states, params), [False, True, False]
    )


def test_resets_use_exact_post_impact_angles_and_momentum_rule():
    params = rimless_wheel.generate_params()
    upper_angle = rimless_wheel.impact_angle(params)
    lower_angle = rimless_wheel.post_impact_angle(params)
    collision_factor = np.cos(2.0 * params["alpha"])
    overshot_state = np.array([upper_angle + 0.03, 2.0])

    reset = rimless_wheel.reset_state(overshot_state, params)

    assert reset[0] == pytest.approx(lower_angle)
    assert reset[1] == pytest.approx(2.0 * collision_factor)

    backward_reset = rimless_wheel.reset_state_backward(
        np.array([lower_angle - 0.03, -2.0]), params
    )
    backward_reset_alias = rimless_wheel.reset_backward_state(
        np.array([lower_angle - 0.03, -2.0]), params
    )
    assert backward_reset[0] == pytest.approx(upper_angle)
    assert backward_reset[1] == pytest.approx(-2.0 * collision_factor)
    np.testing.assert_allclose(backward_reset_alias, backward_reset)


def test_energy_is_conserved_during_swing_and_lost_only_at_impact():
    params = rimless_wheel.generate_params()
    alpha = params["alpha"]
    slope = params["gamma"]
    gravity = params["gravity"]
    length = params["length"]
    collision_factor = np.cos(2.0 * alpha)

    post_impact_speed = 1.5
    swing_angles = np.linspace(slope - alpha, slope + alpha, 7)
    swing_speeds = np.sqrt(
        post_impact_speed**2
        + 2.0 * gravity / length * (np.cos(slope - alpha) - np.cos(swing_angles))
    )
    preimpact_speed = swing_speeds[-1]
    next_post_impact_speed = collision_factor * preimpact_speed
    states = np.array(
        [
            np.append(swing_angles, slope - alpha),
            np.append(swing_speeds, next_post_impact_speed),
        ]
    )

    kinetic, potential = rimless_wheel.calculate_energy(states, params)
    total = kinetic + potential

    np.testing.assert_allclose(total[:-1], total[0])
    assert potential[-2] == pytest.approx(potential[-1])
    assert kinetic[-1] == pytest.approx(collision_factor**2 * kinetic[-2])
    assert total[-1] < total[-2]

    sparse_forward = states[:, [0, -2, -1]]
    sparse_kinetic, sparse_potential = rimless_wheel.calculate_energy(
        sparse_forward, params
    )
    assert sparse_kinetic[0] + sparse_potential[0] == pytest.approx(
        sparse_kinetic[1] + sparse_potential[1]
    )
    assert sparse_potential[1] == pytest.approx(sparse_potential[2])

    backward_post_speed = -2.0
    backward_swing_angles = np.linspace(slope + alpha, slope - alpha, 7)
    backward_swing_speeds = -np.sqrt(
        backward_post_speed**2
        + 2.0
        * gravity
        / length
        * (np.cos(slope + alpha) - np.cos(backward_swing_angles))
    )
    backward_preimpact_speed = backward_swing_speeds[-1]
    backward_states = np.array(
        [
            np.append(backward_swing_angles, slope + alpha),
            np.append(
                backward_swing_speeds,
                collision_factor * backward_preimpact_speed,
            ),
        ]
    )
    backward_kinetic, backward_potential = rimless_wheel.calculate_energy(
        backward_states, params
    )
    backward_total = backward_kinetic + backward_potential
    np.testing.assert_allclose(backward_total[:-1], backward_total[0])
    assert backward_potential[-2] == pytest.approx(backward_potential[-1])
    assert backward_kinetic[-1] == pytest.approx(
        collision_factor**2 * backward_kinetic[-2]
    )
    assert backward_total[-1] < backward_total[-2]

    sparse_backward = backward_states[:, [0, -2, -1]]
    sparse_kinetic, sparse_potential = rimless_wheel.calculate_energy(
        sparse_backward, params
    )
    assert sparse_kinetic[0] + sparse_potential[0] == pytest.approx(
        sparse_kinetic[1] + sparse_potential[1]
    )
    assert sparse_potential[1] == pytest.approx(sparse_potential[2])


def test_default_analytical_gait_values_and_aliases():
    params = rimless_wheel.generate_params()
    alpha = params["alpha"]
    slope = params["gamma"]
    gravity = params["gravity"]
    length = params["length"]
    collision_factor = np.cos(2.0 * alpha)
    squared_speed_gain = 4.0 * gravity / length * np.sin(alpha) * np.sin(slope)

    expected_minimum = np.sqrt(2.0 * gravity / length * (1.0 - np.cos(alpha - slope)))
    expected_fixed_point = collision_factor * np.sqrt(
        squared_speed_gain / (1.0 - collision_factor**2)
    )

    assert rimless_wheel.minimum_crossing_speed(params) == pytest.approx(
        expected_minimum
    )
    assert rimless_wheel.fixed_point_speed(params) == pytest.approx(
        expected_fixed_point
    )
    assert rimless_wheel.fixed_speed(params) == pytest.approx(expected_fixed_point)
    assert rimless_wheel.step_to_step_return_map(
        expected_fixed_point, params
    ) == pytest.approx(expected_fixed_point)
    assert rimless_wheel.return_map(expected_fixed_point, params) == pytest.approx(
        expected_fixed_point
    )
    assert rimless_wheel.floquet_multiplier(params) == pytest.approx(
        collision_factor**2
    )


def test_signed_return_map_covers_rolling_and_standing_branches():
    params = rimless_wheel.generate_params()
    alpha = params["alpha"]
    slope = params["gamma"]
    gravity = params["gravity"]
    length = params["length"]
    collision_factor = np.cos(2.0 * alpha)
    speed_gain = 4.0 * gravity / length * np.sin(alpha) * np.sin(slope)
    uphill_threshold, downhill_threshold = rimless_wheel.return_map_thresholds(params)
    speeds = np.array(
        [
            uphill_threshold - 0.4,
            0.5 * uphill_threshold,
            0.5 * downhill_threshold,
            downhill_threshold + 0.4,
            uphill_threshold,
            downhill_threshold,
        ]
    )

    mapped = rimless_wheel.return_map(speeds, params)

    assert mapped[0] == pytest.approx(
        -collision_factor * np.sqrt(speeds[0] ** 2 - speed_gain)
    )
    assert mapped[1] == pytest.approx(-collision_factor * speeds[1])
    assert mapped[2] == pytest.approx(-collision_factor * speeds[2])
    assert mapped[3] == pytest.approx(
        collision_factor * np.sqrt(speeds[3] ** 2 + speed_gain)
    )
    assert np.isnan(mapped[4])
    assert np.isnan(mapped[5])


def test_no_passive_gait_on_level_ground():
    params = rimless_wheel.generate_params(slope=0.0)

    assert np.isnan(rimless_wheel.fixed_point_speed(params))
    assert np.isnan(rimless_wheel.floquet_multiplier(params))


def test_invalid_or_inconsistent_parameters_are_rejected():
    with pytest.raises(ValueError, match="integer"):
        rimless_wheel.generate_params(number_of_spokes=7.5)
    with pytest.raises(ValueError, match="positive"):
        rimless_wheel.generate_params(length=0.0)

    stale_geometry = rimless_wheel.generate_params()
    stale_geometry["N"] = 10
    with pytest.raises(ValueError, match="alpha"):
        rimless_wheel.need_reset(np.array([0.5, 1.0]), stale_geometry)

    damped = rimless_wheel.generate_params(damping=0.1)
    with pytest.raises(ValueError, match="undamped"):
        rimless_wheel.fixed_point_speed(damped)
