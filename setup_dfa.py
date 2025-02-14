from gurobipy import Model, GRB
from constraints import (add_transition_constraints,
                         add_prefix_constraints,
                         add_initial_state_constraint,
                         add_transition_follow_constraints,
                         add_alpha_constraints,
                         )
from utility import get_prefixes


def initialize_prefixes_and_bounds(sample, alphabet, lower_bound, upper_bound, min_dfa_size, start_token=""):
    """Initialize prefixes, bounds, and validate arguments."""
    if not lower_bound and not upper_bound:
        raise ValueError('Both lower_bound and upper_bound are not set')

    if lower_bound and upper_bound and lower_bound > upper_bound:
        raise ValueError('Invalid setup: lower_bound > upper_bound')

    if (not lower_bound or not upper_bound) and min_dfa_size == 1:
        raise ValueError('Invalid setup: one of lower_bound or upper_bound is not set while min_dfa_size == 1')

    start_token = ""
    prefixes = get_prefixes(sample, start_token)
    max_prefix_size = len(prefixes)
    max_dfa_size = max_prefix_size + 2
    return prefixes, max_dfa_size


def setup_model(n, alphabet, prefixes):
    """Set up the Gurobi model for DFA learning."""
    model = Model(f"DFA_size_{n}")
    states = range(n)

    delta = model.addVars(states, alphabet, states, vtype=GRB.BINARY, name="delta")
    final_states = model.addVars(states, vtype=GRB.BINARY, name="f")
    x = model.addVars(((w, q) for w in prefixes for q in states), vtype=GRB.BINARY, name="x")
    return model, states, delta, final_states, x


def set_dfa(sample, alphabet, dfa_size, prefixes=None, start_token="", verbose=0):
    if prefixes is None:
        prefixes = get_prefixes(sample, start_token)

    model, states, delta, final_states, x = setup_model(dfa_size, alphabet, prefixes)
    model.setParam("OutputFlag", 1 if verbose >= 2 else 0)

    add_transition_constraints(model, states, alphabet, delta)
    add_prefix_constraints(model, states, prefixes, x)
    add_initial_state_constraint(model, x, "")
    add_transition_follow_constraints(model, states, prefixes, alphabet, x, delta)
    alpha = add_alpha_constraints(model, states, sample, x, final_states)

    return model, states, delta, final_states, alpha
