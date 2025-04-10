from typing import Any, List, Optional, Tuple, Union
from gurobipy import GRB, Model, Var, quicksum, tupledict
import numpy as np
from deprecated.distance_functions import find_max_dist


def add_sink_state_penalty(
    model: Model,
    states: List[str],
    transitions: tupledict[Any, Var],
    alphabet: List[str],
    lambda_s: Union[int, float],
    sink_state_index=1,
):
    """Add penalty for transitions not leading to the sink state."""
    sink_state = states[sink_state_index]
    penalty = lambda_s * quicksum(
        1 - transitions[q, a, sink_state] for q in states for a in alphabet
    )
    model.setObjective(model.getObjective() + penalty, GRB.MINIMIZE)


def add_self_loop_penalty(
    model: Model,
    states: List[str],
    transitions: tupledict[Any, Var],
    alphabet: List[str],
    lambda_l: float,
):
    """Add penalty for transitions that are not self-loops."""
    penalty = lambda_l * quicksum(
        transitions[q, a, q0]
        for q in states
        for a in alphabet
        for q0 in states
        if q0 != q
    )
    model.setObjective(model.getObjective() + penalty, GRB.MINIMIZE)


def add_parallel_edge_penalty(
    model: Model,
    states: List[str],
    transitions: tupledict[Any, Var],
    alphabet: List[str],
    lambda_p: float,
    eq: tupledict[Any, Var],
):
    """Add penalty for multiple successor states."""
    for q in states:
        for q0 in states:
            model.addConstr(
                eq[q, q0] <= quicksum(transitions[q, a, q0] for a in alphabet)
            )
            for a in alphabet:
                model.addConstr(eq[q, q0] >= transitions[q, a, q0])
    penalty = lambda_p * quicksum(eq[q, q0] for q in states for q0 in states)
    model.setObjective(model.getObjective() + penalty, GRB.MINIMIZE)


def add_penalties(
    model: Model,
    states: List[str],
    transitions: tupledict[Any, Var],
    alphabet: List[str],
    lambda_s: float,
    lambda_l: float,
    lambda_p: float,
    eq: tupledict[Any, Var],
):
    if lambda_s != 0:
        add_sink_state_penalty(model, states, transitions, alphabet, lambda_s)
    if lambda_l != 0:
        add_self_loop_penalty(model, states, transitions, alphabet, lambda_l)
    if lambda_p != 0:
        add_parallel_edge_penalty(model, states, transitions, alphabet, lambda_p, eq)


def set_bounded_objective(
    model: Model,
    states: List[str],
    sample: List[Tuple[str]],
    alpha: tupledict[Any, Var],
    lower_bound: int,
    upper_bound: int,
    transitions: tupledict[Any, Var],
    alphabet: List[str],
    lambda_s: float = 0,
    lambda_l: float = 0,
    lambda_p: float = 0,
    eq: Optional[Var] = None,
):
    """Set the objective function for the model.

    - model: Gurobi model.
    - sample: List of sequences in the sample.
    - lower_bound lower number of outliers
    - upper_bound upper number of outliers
    """
    if lower_bound and upper_bound:
        model.setObjective(1, GRB.MINIMIZE)
    elif lower_bound:
        model.setObjective(
            quicksum(alpha[w, q] for w in sample for q in states), GRB.MINIMIZE
        )
    elif upper_bound:
        model.setObjective(
            quicksum(alpha[w, q] for w in sample for q in states), GRB.MAXIMIZE
        )
    add_penalties(
        model, states, transitions, alphabet, lambda_s, lambda_l, lambda_p, eq
    )


def add_distance_based_objective(
    model: Model,
    sample: List[Tuple[str]],
    accepted: tupledict[Any, Var],
    gamma: tupledict[Any, Var],
    betta: tupledict[Any, Var],
    distance_matrix: np.ndarray,
    states: List[str],
    transitions: tupledict[Any, Var],
    alphabet: List[str],
    reg_constant: float = 33.8,
    outlier_weight: float = 0.5,
    lambda_s: float = 0,
    lambda_l: float = 0,
    lambda_p: float = 0,
    eq: Optional[Var] = None,
):
    """
    Add an objective to a Gurobi model based on distances between sequences.

    Parameters:
    - model: Gurobi model.
    - sample: List of sequences in the sample.
    - accepted: Dictionary of accepted variables for each sequence index.
    - gamma: Dictionary of gamma variables for sequence pairs.
    - distance_matrix: Precomputed distance matrix (numpy array).
    - reg_constant: regularization of number of outliers
    - outlier_weight: coefficient for forcing model to find more or less outliers
    """
    n = len(sample)
    max_dist = find_max_dist(distance_matrix)

    objective = quicksum(
        (1 - outlier_weight) * betta[sample[i], sample[j]] * distance_matrix[i, j]
        + outlier_weight * gamma[sample[i], sample[j]] * (0 - distance_matrix[i, j])
        for i in range(n)
        for j in range(n)
    ) + quicksum(reg_constant * accepted[sample[i]] for i in range(n))

    model.setObjective(objective, GRB.MINIMIZE)
    add_penalties(
        model, states, transitions, alphabet, lambda_s, lambda_l, lambda_p, eq
    )
