import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assignment_1
from models import rimless_wheel


def test_event_localized_fixed_point_and_floquet_match_analytic_values():
    params = rimless_wheel.generate_params()
    analytic_fixed_point = rimless_wheel.fixed_point_speed(params)
    analytic_multiplier = rimless_wheel.floquet_multiplier(params)

    numerical_fixed_point = assignment_1.find_numerical_fixed_point(
        params, timestep=5e-3
    )
    left, right, centered = assignment_1.estimate_numerical_floquet_multiplier(
        numerical_fixed_point,
        params,
        relative_perturbation=0.01,
        timestep=5e-3,
    )

    assert numerical_fixed_point == pytest.approx(analytic_fixed_point, abs=1e-6)
    assert centered == pytest.approx(analytic_multiplier, abs=2e-5)
    assert left < analytic_multiplier < right


def test_default_full_map_region_of_attraction_and_known_outcomes():
    params = rimless_wheel.generate_params()
    basin = assignment_1.estimate_region_of_attraction(
        params,
        angle_points=41,
        speed_points=61,
        normalized_speed_limit=1.25,
    )

    assert basin.attractors.shape == (61, 41)
    assert basin.rolling_fraction == pytest.approx(0.575, abs=0.002)
    assert basin.unresolved_fraction == pytest.approx(0.0)
    assert basin.angles[0] > rimless_wheel.uphill_impact_angle(params)
    assert basin.angles[-1] < rimless_wheel.impact_angle(params)

    standing = assignment_1.simulate_hybrid_dynamics(
        np.array([rimless_wheel.post_impact_angle(params), 0.5]),
        timestep=5e-3,
        duration=10.0,
        params=params,
    )
    rolling = assignment_1.simulate_hybrid_dynamics(
        np.array([rimless_wheel.post_impact_angle(params), 1.5]),
        timestep=5e-3,
        duration=10.0,
        params=params,
    )

    assert standing.outcome == "standing fixed point"
    assert rolling.outcome == "rolling limit cycle"

    reverse_then_roll = assignment_1.simulate_hybrid_dynamics(
        np.array([rimless_wheel.impact_angle(params), -3.5]),
        timestep=5e-3,
        duration=12.0,
        params=params,
    )
    assert reverse_then_roll.outcome == "rolling limit cycle"
    assert 1 in reverse_then_roll.impact_directions


def test_eight_spoke_critical_slope_marks_gait_onset():
    critical_slope = assignment_1.calculate_critical_slope(8)

    assert critical_slope == pytest.approx(0.06822945245228716)
    below = rimless_wheel.generate_params(slope=critical_slope - 1e-5)
    at_onset = rimless_wheel.generate_params(slope=critical_slope)
    above = rimless_wheel.generate_params(slope=critical_slope + 1e-5)
    assert np.isnan(rimless_wheel.fixed_point_speed(below))
    assert np.isnan(rimless_wheel.fixed_point_speed(at_onset))
    assert np.isfinite(rimless_wheel.fixed_point_speed(above))
