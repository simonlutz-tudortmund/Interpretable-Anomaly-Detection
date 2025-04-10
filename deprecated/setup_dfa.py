from typing import List, Optional, Tuple
from gurobipy import Model, GRB
from deprecated.constraints import (
    add_transition_constraints,
    add_prefix_constraints,
    add_initial_state_constraint,
    add_transition_follow_constraints,
    add_alpha_constraints,
)
from deprecated.utility import get_prefixes


def initialize_prefixes_and_bounds(
    sample: List[Tuple[str]],
    alphabet: List[str],
    lower_bound: int,
    upper_bound: int,
    min_dfa_size: int,
    start_token: str = "",
):
    """Initialize prefixes, bounds, and validate arguments."""
    if not lower_bound and not upper_bound:
        raise ValueError("Both lower_bound and upper_bound are not set")

    if lower_bound and upper_bound and lower_bound > upper_bound:
        raise ValueError("Invalid setup: lower_bound > upper_bound")

    if (not lower_bound or not upper_bound) and min_dfa_size == 1:
        raise ValueError(
            "Invalid setup: one of lower_bound or upper_bound is not set while min_dfa_size == 1"
        )

    start_token = ""
    prefixes = get_prefixes(sample, start_token)
    max_prefix_size = len(prefixes)
    max_dfa_size = max_prefix_size + 2
    return prefixes, max_dfa_size


def setup_model(n: int, alphabet: List[str], prefixes: List[str]):
    """Set up the Gurobi model for DFA learning."""
    model = Model(f"DFA_size_{n}")
    states = list(range(n))

    transitions = model.addVars(
        states, alphabet, states, vtype=GRB.BINARY, name="transitions"
    )
    final_states = model.addVars(states, vtype=GRB.BINARY, name="f")
    x = model.addVars(((w, q) for w in prefixes for q in states), vtype=GRB.BINARY)
    return model, states, transitions, final_states, x


def set_dfa(
    sample: List[Tuple[str]],
    alphabet: List[str],
    dfa_size: int,
    prefixes: Optional[List[str]] = None,
    start_token: str = "",
    verbose: int = 0,
):
    if prefixes is None:
        prefixes = get_prefixes(sample, start_token)

    model, states, transitions, final_states, x = setup_model(
        n=dfa_size, alphabet=alphabet, prefixes=prefixes
    )
    model.setParam("OutputFlag", 1 if verbose >= 2 else 0)

    add_transition_constraints(
        model=model, states=states, alphabet=alphabet, transitions=transitions
    )
    add_prefix_constraints(model=model, states=states, prefixes=prefixes, x=x)
    add_initial_state_constraint(model=model, x=x, start_token="")
    add_transition_follow_constraints(
        model=model,
        states=states,
        prefixes=prefixes,
        alphabet=alphabet,
        x=x,
        transitions=transitions,
    )
    alpha = add_alpha_constraints(
        model=model, states=states, sample=sample, x=x, final_states=final_states
    )

    return model, states, transitions, final_states, alpha
