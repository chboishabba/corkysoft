import math

import pytest

from profit_optimizer import ProfitOptimizer


def test_optimal_job_mix_with_binding_constraints():
    optimizer = ProfitOptimizer()
    optimizer.add_variable("local_move", profit_per_unit=300.0, upper_bound=40)
    optimizer.add_variable("interstate_move", profit_per_unit=500.0)

    optimizer.add_constraint(
        "crew_hours",
        coefficients={"local_move": 2.0, "interstate_move": 3.0},
        rhs=120.0,
    )
    optimizer.add_constraint(
        "truck_days",
        coefficients={"local_move": 1.0, "interstate_move": 2.0},
        rhs=80.0,
    )

    result = optimizer.solve()

    assert result.status == "optimal"
    assert math.isclose(result.objective_value, 20000.0, rel_tol=1e-6)
    assert math.isclose(result.variable_values["local_move"], 0.0, abs_tol=1e-8)
    assert math.isclose(result.variable_values["interstate_move"], 40.0, abs_tol=1e-8)

    # Both operational constraints are fully utilised and therefore binding.
    assert set(result.binding_constraints) == {"crew_hours", "truck_days"}

    # The slack for the automatic upper-bound constraint equals the remaining capacity.
    assert math.isclose(result.slack_values["slack_local_move_ub"], 40.0, abs_tol=1e-8)

    # Reduced cost for the non-basic variable is negative, confirming it would reduce profit if increased.
    assert result.reduced_costs["local_move"] < 0


def test_detects_unbounded_problem():
    optimizer = ProfitOptimizer()
    optimizer.add_variable("runway_lane", profit_per_unit=100.0)

    # Without constraints (or bounds) the problem is unbounded.
    with pytest.raises(ValueError):
        optimizer.solve()

    optimizer.add_constraint("demand_cap", coefficients={"runway_lane": -1.0}, rhs=-10.0)

    # After normalisation the constraint becomes runway_lane <= 10, giving a bounded solution.
    result = optimizer.solve()
    assert math.isclose(result.variable_values["runway_lane"], 10.0, abs_tol=1e-8)
    assert math.isclose(result.objective_value, 1000.0, abs_tol=1e-8)
    assert result.binding_constraints == ["demand_cap"]
