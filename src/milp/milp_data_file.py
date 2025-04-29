#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import dis
import itertools
import logging
from math import ceil, floor
from typing import Literal, Optional, Tuple

from gurobipy import GRB
import numpy as np

from src.utils.distance_functions import (
    calculate_distance_matrix,
    distance_function_names,
)
from src.milp.problem import Problem
from utils.dfa import CausalDFA


def learn_dfa_with_bounds(
    sample: list[tuple[str, ...]],
    alphabet: frozenset[str],
    lower_bound: float,
    upper_bound: float,
    lambda_s: Optional[float],
    lambda_l: Optional[float],
    lambda_p: Optional[float],
    min_dfa_size: int,
    verbose=0,
) -> Tuple[CausalDFA, Problem]:
    """Trains a DFA using the specified algorithm and returns the learned DFA."""

    if lower_bound and not lower_bound.is_integer():
        lower_bound = ceil(lower_bound * len(sample))
    if upper_bound and not upper_bound.is_integer():
        upper_bound = floor(upper_bound * len(sample))

    if lower_bound:
        lower_bound = int(lower_bound)
    if upper_bound:
        upper_bound = int(upper_bound)

    unique_samples = []
    sample_counts = {}
    """Add constraints for prefix acceptance."""
    for word in sample:
        # Convert the list to a tuple to make it hashable
        word_tuple = tuple(word)
        if word_tuple in sample_counts:
            sample_counts[word_tuple] += 1
        else:
            sample_counts[word_tuple] = 1
            unique_samples.append(word)
    sample_freq = np.array([sample_counts[tuple(word)] for word in unique_samples])

    max_size = None
    for N in itertools.count(min_dfa_size):
        if max_size is not None and N > max_size:
            raise RuntimeError("DFA with at most {} states not found".format(max_size))
            # return None

        logging.warning(
            f"\x1b[1;34m>>> trying to solve DFA with {N} states and bounds [{lower_bound}, {upper_bound}].\x1b[m"
        )

        problem = Problem(N, alphabet)
        problem.model.setParam("Threads", 1)

        # Aggressive presolve + relaxation heuristic
        problem.model.setParam("Presolve", 2)
        # if len(sample) > 1000:
        #     problem.model.setParam("NoRelHeurTime", 1800)
        # problem.model.setParam("Cuts", 0)

        # Avoid Simplex altogether
        problem.model.setParam("Method", 2)
        problem.model.setParam("Crossover", 0)
        problem.model.setParam("NodeMethod", 2)

        problem.model.setParam("OutputFlag", 1 if verbose >= 2 else 0)

        problem.add_automaton_constraints()
        problem.add_unlabeled_sample_constraints(
            sample=unique_samples,
            sample_freq=sample_freq,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        if lambda_l:
            problem.add_self_loop_penalty(lambda_l=lambda_l)
        if lambda_p:
            problem.add_parallel_edge_penalty(lambda_p=lambda_p)
        if lambda_s:
            problem.add_sink_state_penalty(lambda_s=lambda_s)

        problem.model.update()
        logging.warning(
            "\x1b[1;34m... translated to {} constraints of {} vars.\x1b[m".format(
                problem.model.NumVars, problem.model.NumConstrs
            )
        )

        problem.model.optimize()
        if problem.model.status == GRB.OPTIMAL:
            logging.warning("\x1b[1;34m... optimal.\x1b[m")
            dfa = problem.get_automaton()
            return dfa, problem
    raise RuntimeError("DFA with at most {} states not found".format(max_size))


def learn_dfa_with_labels(
    positive_sample: list[tuple[str, ...]],
    negative_sample: list[tuple[str, ...]],
    alphabet: frozenset[str],
    lambda_s: Optional[float],
    lambda_l: Optional[float],
    lambda_p: Optional[float],
    min_dfa_size: int,
    verbose=0,
) -> Tuple[CausalDFA, Problem]:
    max_size = None
    for N in itertools.count(min_dfa_size):
        if max_size is not None and N > max_size:
            raise RuntimeError("DFA with at most {} states not found".format(max_size))
            # return None

        logging.warning(
            f"\x1b[1;34m>>> trying to solve DFA with {N} states and {len(positive_sample)} positive, {len(negative_sample)} negative samples.\x1b[m"
        )

        positive_sample = list(set(positive_sample))
        negative_sample = list(set(negative_sample))

        problem = Problem(N, alphabet)
        problem.model.setParam("Threads", 1)

        # Aggressive presolve + relaxation heuristic
        problem.model.setParam("Presolve", 2)
        problem.model.setParam("NoRelHeurTime", 1800)
        problem.model.setParam("Cuts", 0)

        # Avoid Simplex altogether
        problem.model.setParam("Method", 2)
        problem.model.setParam("Crossover", 0)
        problem.model.setParam("NodeMethod", 2)

        problem.model.setParam("OutputFlag", 1 if verbose >= 2 else 0)

        problem.add_automaton_constraints()
        problem.add_labeled_sample_constraints(
            positive_sample=positive_sample, negative_sample=negative_sample
        )
        if lambda_l:
            problem.add_self_loop_penalty(lambda_l=lambda_l)
        if lambda_p:
            problem.add_parallel_edge_penalty(lambda_p=lambda_p)
        if lambda_s:
            problem.add_sink_state_penalty(lambda_s=lambda_s)

        problem.model.update()
        logging.warning(
            "\x1b[1;34m... translated to {} constraints of {} vars.\x1b[m".format(
                problem.model.NumVars, problem.model.NumConstrs
            )
        )

        problem.model.optimize()
        if problem.model.status == GRB.OPTIMAL:
            logging.warning("\x1b[1;34m... optimal.\x1b[m")
            dfa = problem.get_automaton()
            return dfa, problem
    raise RuntimeError("DFA with at most {} states not found".format(max_size))


def learn_dfa_with_distances_linear(
    sample: list[tuple[str, ...]],
    alphabet: frozenset[str],
    distance_function: Literal[
        "levenshtein", "jaccard", "hamming", "ordinal_hamming", "dynamic_time_warping"
    ],
    lambda_s: Optional[float],
    lambda_l: Optional[float],
    lambda_p: Optional[float],
    dfa_size: int,
    verbose=0,
) -> Tuple[CausalDFA, Problem]:
    """Trains a DFA using the specified algorithm and returns the learned DFA."""
    N = dfa_size
    unique_samples = []
    sample_counts = {}
    """Add constraints for prefix acceptance."""
    for word in sample:
        # Convert the list to a tuple to make it hashable
        word_tuple = tuple(word)
        if word_tuple in sample_counts:
            sample_counts[word_tuple] += 1
        else:
            sample_counts[word_tuple] = 1
            unique_samples.append(word)
    sample_freq = np.array(
        [sample_counts[tuple(word)] for word in unique_samples]
    ).reshape(-1, 1)
    sample_freq = sample_freq / np.sum(sample_freq)
    sample_pair_freq = sample_freq @ sample_freq.T

    distance_function_callable = distance_function_names[distance_function]
    distance_matrix = calculate_distance_matrix(
        sample=unique_samples, distance_func=distance_function_callable
    )
    logging.warning(
        f"\x1b[1;34m>>> trying to solve DFA with {N} states and {distance_function} distance function.\x1b[m"
    )

    problem = Problem(N, alphabet)
    problem.model.setParam("Threads", 1)

    # Aggressive presolve + relaxation heuristic
    # problem.model.setParam("DegenMoves", 0)
    problem.model.setParam("Presolve", 2)
    # problem.model.setParam("Cuts", 0)

    # # Avoid Simplex altogether
    # problem.model.setParam("Method", 2)
    # problem.model.setParam("Crossover", 0)
    # problem.model.setParam("NodeMethod", 2)

    problem.model.setParam("OutputFlag", 1 if verbose >= 2 else 0)

    problem.add_automaton_constraints()
    problem.add_distance_sample_constraints_linear(
        sample=unique_samples,
        sample_pair_freq_matrix=sample_pair_freq,
        distance_matrix=distance_matrix,
    )
    if lambda_l:
        problem.add_self_loop_penalty(lambda_l=lambda_l)
    if lambda_p:
        problem.add_parallel_edge_penalty(lambda_p=lambda_p)
    if lambda_s:
        problem.add_sink_state_penalty(lambda_s=lambda_s)

    problem.model.update()
    logging.warning(
        "\x1b[1;34m... translated to {} constraints of {} vars.\x1b[m".format(
            problem.model.NumVars, problem.model.NumConstrs
        )
    )

    problem.model.optimize()
    if problem.model.status == GRB.OPTIMAL:
        logging.warning("\x1b[1;34m... optimal.\x1b[m")
        dfa = problem.get_automaton()
        return dfa, problem
    raise RuntimeError("DFA with {} states not found".format(N))


def learn_dfa_with_distances_quadratic(
    sample: list[tuple[str, ...]],
    alphabet: frozenset[str],
    distance_function: Literal[
        "levenshtein", "jaccard", "hamming", "ordinal_hamming", "dynamic_time_warping"
    ],
    lambda_s: Optional[float],
    lambda_l: Optional[float],
    lambda_p: Optional[float],
    dfa_size: int,
    verbose=0,
) -> Tuple[CausalDFA, Problem]:
    """Trains a DFA using the specified algorithm and returns the learned DFA."""
    N = dfa_size
    unique_samples = []
    sample_counts = {}
    """Add constraints for prefix acceptance."""
    for word in sample:
        # Convert the list to a tuple to make it hashable
        word_tuple = tuple(word)
        if word_tuple in sample_counts:
            sample_counts[word_tuple] += 1
        else:
            sample_counts[word_tuple] = 1
            unique_samples.append(word)
    sample_freq = np.array(
        [sample_counts[tuple(word)] for word in unique_samples]
    ).reshape(-1, 1)
    sample_freq = sample_freq / np.sum(sample_freq)
    sample_pair_freq = sample_freq @ sample_freq.T

    distance_function_callable = distance_function_names[distance_function]
    distance_matrix = calculate_distance_matrix(
        sample=unique_samples, distance_func=distance_function_callable
    )
    logging.warning(
        f"\x1b[1;34m>>> trying to solve DFA with {N} states and {distance_function} distance function.\x1b[m"
    )

    problem = Problem(N, alphabet)
    problem.model.setParam("Threads", 1)

    # Aggressive presolve + relaxation heuristic
    problem.model.setParam("Presolve", 2)
    # if len(sample) > 1000:
    #     problem.model.setParam("NoRelHeurTime", 1800)
    problem.model.setParam("Cuts", 0)

    # Avoid Simplex altogether
    problem.model.setParam("Method", 2)
    problem.model.setParam("Crossover", 0)
    problem.model.setParam("NodeMethod", 2)

    problem.model.setParam("OutputFlag", 1 if verbose >= 2 else 0)

    problem.add_automaton_constraints()
    problem.add_distance_sample_constraints_quadratic(
        sample=unique_samples,
        sample_pair_freq_matrix=sample_pair_freq,
        distance_matrix=distance_matrix,
    )
    if lambda_l:
        problem.add_self_loop_penalty(lambda_l=lambda_l)
    if lambda_p:
        problem.add_parallel_edge_penalty(lambda_p=lambda_p)
    if lambda_s:
        problem.add_sink_state_penalty(lambda_s=lambda_s)

    problem.model.update()
    logging.warning(
        "\x1b[1;34m... translated to {} constraints of {} vars.\x1b[m".format(
            problem.model.NumVars, problem.model.NumConstrs
        )
    )

    problem.model.optimize()
    if problem.model.status == GRB.OPTIMAL:
        logging.warning("\x1b[1;34m... optimal.\x1b[m")
        dfa = problem.get_automaton()
        return dfa, problem
    raise RuntimeError("DFA with {} states not found".format(N))
