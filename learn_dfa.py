from gurobipy import Model, GRB
from distance_functions import calculate_distance_matrix
from constraints import (add_transition_constraints,
                         add_prefix_constraints,
                         add_initial_state_constraint,
                         add_transition_follow_constraints,
                         add_alpha_constraints,
                         add_sample_constraints,
                         add_accepted_constraints,
                         add_gamma_constraints,
                         add_betta_constraints
                         )
from objective_functions import set_bounded_objective, add_distance_based_objective
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


def set_dfa(sample, alphabet, dfa_size, prefixes=None, start_token=""):
    if prefixes is None:
        prefixes = get_prefixes(sample, start_token)

    model, states, delta, final_states, x = setup_model(dfa_size, alphabet, prefixes)
    add_transition_constraints(model, states, alphabet, delta)
    add_prefix_constraints(model, states, prefixes, x)
    add_initial_state_constraint(model, x, "")
    add_transition_follow_constraints(model, states, prefixes, alphabet, x, delta)
    alpha = add_alpha_constraints(model, states, sample, x, final_states)

    return model, states, delta, final_states, alpha


def learn_minimal_dfa(sample, alphabet, lower_bound, upper_bound, min_dfa_size=1, start_token=""):
    prefixes, max_dfa_size = initialize_prefixes_and_bounds(sample,
                                                            alphabet,
                                                            lower_bound,
                                                            upper_bound,
                                                            min_dfa_size,
                                                            start_token)
    for dfa_size in range(min_dfa_size, max_dfa_size):
        model, states, delta, final_states, alpha = set_dfa(sample, alphabet, dfa_size, prefixes, start_token)
        add_sample_constraints(model, states, sample, alpha, lower_bound, upper_bound)
        set_bounded_objective(model, states, sample, alpha, lower_bound, upper_bound)

        model.optimize()
        solution = extract_dfa_solution(model, states, alphabet, delta, final_states)
        if solution:
            return solution
    return None


def learn_distance_based_dfa(sample, alphabet, dfa_size, distance_function, outlier_weight=0.5, start_token=""):
    prefixes = get_prefixes(sample, start_token)
    model, states, delta, final_states, alpha = set_dfa(sample, alphabet, dfa_size, prefixes, start_token)
    accepted = add_accepted_constraints(model, states, sample, alpha)

    gamma = add_gamma_constraints(model, sample, accepted)
    betta = add_betta_constraints(model, sample, accepted)

    distance_matrix = calculate_distance_matrix(sample, distance_function)
    add_distance_based_objective(model, sample, accepted, gamma, betta, distance_matrix, outlier_weight)

    model.optimize()

    solution = extract_dfa_solution(model, states, alphabet, delta, final_states)
    if solution:
        return solution


def extract_dfa_solution(model, states, alphabet, delta, final_states):
    """Extract the DFA solution from the model."""
    if model.status == GRB.OPTIMAL:
        dfa_transitions = {(q, a): q_prime for q in states for a in alphabet for q_prime in states
                           if delta[q, a, q_prime].x > 0.5}
        final_states_set = {q for q in states if final_states[q].x > 0.5}
        return {
            "states": states,
            "alphabet": alphabet,
            "transitions": dfa_transitions,
            "initial_state": 0,
            "final_states": final_states_set
        }
    return None