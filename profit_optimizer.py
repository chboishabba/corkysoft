"""Simplex-based profit optimization utilities.

This module provides a light-weight linear programming solver focussed on
transport job planning problems.  Jobs or lanes are modelled as decision
variables with an associated profit contribution, while business rules such as
available truck hours, cubic metre capacity, or market demand are represented
as linear constraints.  The solver applies the (primal) Simplex algorithm to
recommend an optimal mix that maximises total profit.

Only non-negative decision variables and ``<=`` style constraints are
supported.  This matches the typical "allocate capacity" framing where each job
fraction is greater than or equal to zero and cannot exceed a resource limit.
Upper bounds for individual variables can be expressed either through explicit
constraints or directly when adding a variable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import math


@dataclass
class DecisionVariable:
    """A decision variable in the linear program."""

    name: str
    profit: float
    index: int
    upper_bound: Optional[float] = None


@dataclass
class Constraint:
    """A linear ``<=`` constraint defined over decision variables."""

    name: str
    coefficients: Mapping[str, float]
    rhs: float

    def normalised(self) -> "Constraint":
        """Return a constraint with a non-negative right-hand side.

        Simplex requires constraints in the form ``Ax <= b`` with ``b >= 0``. If
        ``b`` is negative we multiply the entire row by ``-1`` which produces an
        equivalent constraint that still respects the ``<=`` orientation.
        """

        if self.rhs >= 0:
            return self
        flipped = {var: -coeff for var, coeff in self.coefficients.items()}
        return Constraint(name=self.name, coefficients=flipped, rhs=-self.rhs)


@dataclass
class OptimizationResult:
    """Container for the Simplex optimisation outcome."""

    status: str
    objective_value: float
    variable_values: Dict[str, float]
    slack_values: Dict[str, float]
    binding_constraints: List[str]
    reduced_costs: Dict[str, float]
    iterations: int = 0

    def __post_init__(self) -> None:  # pragma: no cover - defensive programming
        # Normalise tiny floating point artefacts to zero for nicer consumption.
        for values in (self.variable_values, self.slack_values, self.reduced_costs):
            for key, value in list(values.items()):
                if abs(value) < 1e-12:
                    values[key] = 0.0


class SimplexUnboundedError(RuntimeError):
    """Raised when the Simplex pivot cannot be computed due to unboundedness."""


class ProfitOptimizer:
    """Simplex-based optimiser for maximising lane or job profitability.

    Typical usage::

        optimizer = ProfitOptimizer()
        optimizer.add_variable("lane_a", profit_per_unit=1200)
        optimizer.add_variable("lane_b", profit_per_unit=800, upper_bound=30)
        optimizer.add_constraint(
            "truck_hours",
            coefficients={"lane_a": 3.0, "lane_b": 2.0},
            rhs=180,
        )
        result = optimizer.solve()

    The resulting :class:`OptimizationResult` exposes the optimal mix,
    constraint slacks, binding constraint names, and reduced costs for
    sensitivity analysis.
    """

    def __init__(self) -> None:
        self._variables: List[DecisionVariable] = []
        self._var_index: Dict[str, int] = {}
        self._constraints: List[Constraint] = []

    # ------------------------------------------------------------------
    # Variable & constraint construction helpers
    # ------------------------------------------------------------------
    def add_variable(
        self,
        name: str,
        profit_per_unit: float,
        *,
        upper_bound: Optional[float] = None,
    ) -> None:
        """Register a new decision variable.

        Parameters
        ----------
        name:
            Unique identifier for the variable.
        profit_per_unit:
            Objective coefficient representing contribution to profit per unit
            of the decision variable.
        upper_bound:
            Optional ``<=`` style upper bound. When provided the optimiser will
            automatically introduce an auxiliary constraint so users do not
            need to add one manually.
        """

        if name in self._var_index:
            raise ValueError(f"Variable '{name}' already exists")
        index = len(self._variables)
        self._variables.append(
            DecisionVariable(name=name, profit=profit_per_unit, index=index, upper_bound=upper_bound)
        )
        self._var_index[name] = index

    def add_constraint(
        self, name: str, *, coefficients: Mapping[str, float], rhs: float
    ) -> None:
        """Add a ``<=`` constraint to the optimisation problem.

        Parameters
        ----------
        name:
            Descriptive identifier. This is used when reporting binding
            constraints.
        coefficients:
            Mapping from variable name to its coefficient in the constraint.
            Variables omitted are implicitly treated as having a zero
            coefficient.
        rhs:
            Right-hand side constant. A negative ``rhs`` is automatically
            normalised by multiplying the entire constraint by ``-1`` so that it
            remains compatible with Simplex standard form.
        """

        unknown = set(coefficients) - set(self._var_index)
        if unknown:
            missing = ", ".join(sorted(unknown))
            raise ValueError(f"Constraint '{name}' references unknown variables: {missing}")
        self._constraints.append(Constraint(name=name, coefficients=dict(coefficients), rhs=rhs))

    # ------------------------------------------------------------------
    # Simplex solver
    # ------------------------------------------------------------------
    def solve(self, *, tolerance: float = 1e-9, max_iterations: int = 10_000) -> OptimizationResult:
        """Execute the Simplex algorithm and return the optimal solution."""

        if not self._variables:
            raise ValueError("At least one decision variable is required")
        if not self._constraints and all(v.upper_bound is None for v in self._variables):
            raise ValueError("The problem is unbounded without any constraints")

        variables = list(self._variables)
        constraints = [constraint.normalised() for constraint in self._constraints]

        # Inject upper-bound constraints if necessary.
        auto_constraints: List[Constraint] = []
        for variable in variables:
            if variable.upper_bound is not None:
                auto_constraints.append(
                    Constraint(
                        name=f"{variable.name}_ub",
                        coefficients={variable.name: 1.0},
                        rhs=variable.upper_bound,
                    )
                )
        constraints.extend(auto_constraints)

        num_vars = len(variables)
        num_constraints = len(constraints)
        if num_constraints == 0:
            raise ValueError("No constraints available to bound the optimisation")

        tableau = _initial_tableau(variables, constraints)
        basis = list(range(num_vars, num_vars + num_constraints))

        iterations = 0
        while True:
            iterations += 1
            if iterations > max_iterations:
                raise RuntimeError("Maximum number of Simplex iterations exceeded")

            pivot_col = _choose_entering_variable(tableau, tolerance)
            if pivot_col is None:
                break  # Optimal solution reached.

            pivot_row = _choose_leaving_variable(tableau, pivot_col, tolerance)
            if pivot_row is None:
                raise SimplexUnboundedError("Linear program is unbounded")

            _pivot(tableau, pivot_row, pivot_col)
            basis[pivot_row] = pivot_col

        variable_values = _extract_variable_values(tableau, basis, variables)
        slack_values = _extract_slack_values(tableau, basis, variables, constraints)
        binding_constraints = _identify_binding_constraints(constraints, variable_values, tolerance)
        reduced_costs = _reduced_costs(tableau, basis, variables, tolerance)
        objective_value = tableau[-1, -1]

        return OptimizationResult(
            status="optimal",
            objective_value=objective_value,
            variable_values=variable_values,
            slack_values=slack_values,
            binding_constraints=binding_constraints,
            reduced_costs=reduced_costs,
            iterations=iterations,
        )


# ----------------------------------------------------------------------
# Simplex helpers
# ----------------------------------------------------------------------

def _initial_tableau(variables: Sequence[DecisionVariable], constraints: Sequence[Constraint]):
    import numpy as np

    num_vars = len(variables)
    num_constraints = len(constraints)
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1), dtype=float)

    var_index = {variable.name: variable.index for variable in variables}

    for row, constraint in enumerate(constraints):
        rhs = constraint.rhs
        if math.isinf(rhs) or math.isnan(rhs):
            raise ValueError(f"Constraint '{constraint.name}' has invalid rhs: {rhs}")
        if rhs < 0:  # ``normalised`` should have prevented this.
            raise ValueError(f"Constraint '{constraint.name}' has negative rhs after normalisation")
        for name, coefficient in constraint.coefficients.items():
            tableau[row, var_index[name]] = coefficient
        slack_col = num_vars + row
        tableau[row, slack_col] = 1.0
        tableau[row, -1] = rhs

    for variable in variables:
        tableau[-1, variable.index] = -variable.profit

    return tableau


def _choose_entering_variable(tableau, tolerance: float) -> Optional[int]:
    import numpy as np

    objective_row = tableau[-1, :-1]
    min_value = np.min(objective_row)
    if min_value >= -tolerance:
        return None
    return int(np.argmin(objective_row))


def _choose_leaving_variable(tableau, pivot_col: int, tolerance: float) -> Optional[int]:
    import numpy as np

    column = tableau[:-1, pivot_col]
    rhs = tableau[:-1, -1]
    ratios = []
    for i, (col_value, rhs_value) in enumerate(zip(column, rhs)):
        if col_value > tolerance:
            ratios.append((rhs_value / col_value, i))
    if not ratios:
        return None
    # Choose the minimum ratio (ties resolved by the earliest index which helps prevent cycling).
    ratios.sort()
    return ratios[0][1]


def _pivot(tableau, pivot_row: int, pivot_col: int) -> None:
    import numpy as np

    pivot_element = tableau[pivot_row, pivot_col]
    tableau[pivot_row, :] = tableau[pivot_row, :] / pivot_element
    for row in range(tableau.shape[0]):
        if row == pivot_row:
            continue
        factor = tableau[row, pivot_col]
        tableau[row, :] -= factor * tableau[pivot_row, :]


def _extract_variable_values(tableau, basis: Sequence[int], variables: Sequence[DecisionVariable]) -> Dict[str, float]:
    values = {variable.name: 0.0 for variable in variables}
    for row_index, column_index in enumerate(basis):
        if column_index < len(variables):
            values[variables[column_index].name] = tableau[row_index, -1]
    return values


def _extract_slack_values(
    tableau,
    basis: Sequence[int],
    variables: Sequence[DecisionVariable],
    constraints: Sequence[Constraint],
) -> Dict[str, float]:
    num_vars = len(variables)
    slack_names = [f"slack_{constraint.name}" for constraint in constraints]
    values = {name: 0.0 for name in slack_names}
    for row_index, column_index in enumerate(basis):
        if column_index >= num_vars:
            slack_name = slack_names[column_index - num_vars]
            values[slack_name] = tableau[row_index, -1]
    return values


def _identify_binding_constraints(
    constraints: Sequence[Constraint],
    variable_values: Mapping[str, float],
    tolerance: float,
) -> List[str]:
    binding: List[str] = []
    for constraint in constraints:
        lhs = 0.0
        for name, coefficient in constraint.coefficients.items():
            lhs += coefficient * variable_values.get(name, 0.0)
        slack = constraint.rhs - lhs
        if abs(slack) <= tolerance:
            binding.append(constraint.name)
    return binding


def _reduced_costs(tableau, basis: Sequence[int], variables: Sequence[DecisionVariable], tolerance: float) -> Dict[str, float]:
    import numpy as np

    basic_columns = set(index for index in basis if index < len(variables))
    reduced = {}
    for variable in variables:
        value = tableau[-1, variable.index]
        if variable.index in basic_columns and abs(value) <= tolerance:
            # Basic variables have a reduced cost of zero.
            continue
        reduced_cost = -value
        if abs(reduced_cost) <= tolerance:
            reduced_cost = 0.0
        reduced[variable.name] = reduced_cost
    return reduced


__all__ = [
    "ProfitOptimizer",
    "OptimizationResult",
    "DecisionVariable",
    "Constraint",
    "SimplexUnboundedError",
]
