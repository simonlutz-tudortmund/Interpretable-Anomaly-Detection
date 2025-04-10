import logging
from typing import List, Tuple
from gurobipy import GRB, Model
from src.distance_functions import calculate_distance_matrix
from src.constraints import (
    add_unlabeled_sample_constraints,
    add_accepted_constraints,
    add_gamma_constraints,
    add_betta_constraints,
    add_interpretability_constrates,
)
from src.objective_functions import set_bounded_objective, add_distance_based_objective
from src.utility import get_prefixes
from src.setup_dfa import initialize_prefixes_and_bounds, set_dfa


def fit_minimal_dfa(
    sample: List[Tuple[str]],
    alphabet: List[str],
    lower_bound: int,
    upper_bound: int,
    lambda_s: float,
    lambda_l: float,
    lambda_p: float,
    min_dfa_size=1,
    start_token="",
    verbose=0,
    pointwise=False,
):
    prefixes, max_dfa_size = initialize_prefixes_and_bounds(
        sample, alphabet, lower_bound, upper_bound, min_dfa_size, start_token
    )

    if pointwise:
        sample = prefixes

    for dfa_size in range(min_dfa_size, max_dfa_size):
        logging.warning(
            f"\x1b[1;35m Trying to solve DFA with {dfa_size} states  \x1b[m"
        )
        model, states, transitions, final_states, alpha = set_dfa(
            sample, alphabet, dfa_size, prefixes, start_token, verbose
        )

        eq = add_interpretability_constrates(
            model, states, alphabet, transitions, final_states, lambda_s, lambda_p
        )

        add_unlabeled_sample_constraints(
            model, states, sample, alpha, lower_bound, upper_bound
        )
        set_bounded_objective(
            model,
            states,
            sample,
            alpha,
            lower_bound,
            upper_bound,
            transitions,
            alphabet,
            lambda_s,
            lambda_l,
            lambda_p,
            eq,
        )
        model.update()

        logging.warning(
            "\x1b[1;34m... translated to {} constraints of {} vars.\x1b[m".format(
                model.NumVars, model.NumConstrs
            )
        )
        model.optimize()
        solution = extract_dfa_solution(
            model, states, alphabet, transitions, final_states
        )
        if solution:
            return solution
    return None


def fit_distance_based_dfa(
    sample,
    alphabet,
    dfa_size,
    distance_function,
    outlier_weight=0.5,
    lambda_s=0,
    lambda_l=0,
    lambda_p=0,
    start_token="",
    verbose=0,
    pointwise=False,
):
    prefixes = get_prefixes(sample, start_token)

    if pointwise:
        sample = prefixes

    model, states, transitions, final_states, alpha = set_dfa(
        sample, alphabet, dfa_size, prefixes, start_token, verbose
    )
    accepted = add_accepted_constraints(model, states, sample, alpha)

    gamma = add_gamma_constraints(model, sample, accepted)
    betta = add_betta_constraints(model, sample, accepted)

    eq = add_interpretability_constrates(
        model, states, alphabet, transitions, final_states, lambda_s, lambda_p
    )

    distance_matrix = calculate_distance_matrix(sample, distance_function)
    add_distance_based_objective(
        model,
        sample,
        accepted,
        gamma,
        betta,
        distance_matrix,
        states,
        transitions,
        alphabet,
        outlier_weight=outlier_weight,
        lambda_s=lambda_s,
        lambda_l=lambda_l,
        lambda_p=lambda_p,
        eq=eq,
    )

    model.optimize()

    solution = extract_dfa_solution(model, states, alphabet, transitions, final_states)
    if solution:
        return solution


def extract_dfa_solution(
    model: Model, states: List[str], alphabet, transitions, final_states
):
    """Extract the DFA solution from the model."""
    if model.status == GRB.OPTIMAL:
        dfa_transitions = {
            (q, a): q_prime
            for q in states
            for a in alphabet
            for q_prime in states
            if transitions[q, a, q_prime].x > 0.5
        }
        final_states_set = {q for q in states if final_states[q].x > 0.5}
        return {
            "states": states,
            "alphabet": alphabet,
            "transitions": dfa_transitions,
            "initial_state": 0,
            "final_states": final_states_set,
        }
    return None
