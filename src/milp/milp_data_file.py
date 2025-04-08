#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import logging
from math import ceil, floor

from gurobipy import GRB

from src.milp.problem import Problem


def learn_dfa(
    sample: list[tuple[str, ...]],
    alphabet: frozenset[str],
    lower_bound: float,
    upper_bound: float,
    lambda_s: float,
    lambda_l: float,
    lambda_p: float,
    min_dfa_size: int,
    verbose=0,
    pointwise=False,
):
    """Trains a DFA using the specified algorithm and returns the learned DFA."""

    if lower_bound and not lower_bound.is_integer():
        lower_bound = ceil(lower_bound * len(sample))
    if upper_bound and not upper_bound.is_integer():
        upper_bound = floor(upper_bound * len(sample))

    if lower_bound:
        lower_bound = int(lower_bound)
    if upper_bound:
        upper_bound = int(upper_bound)

    max_size = None
    for N in itertools.count(min_dfa_size):
        if max_size is not None and N > max_size:
            raise RuntimeError("DFA with at most {} states not found".format(max_size))
            # return None

        logging.warning(f"\x1b[1m>>> trying to solve DFA with {N} states.\x1b[m")

        problem = Problem(N, alphabet)
        problem.model.setParam("Threads", 1)
        problem.model.setParam("Method", 0)
        problem.model.setParam("OutputFlag", 1 if verbose >= 2 else 0)

        problem.add_sample_constraints(
            sample=sample, lower_bound=lower_bound, upper_bound=upper_bound
        )
        problem.add_parallel_edge_penalty(lambda_p=lambda_p)
        problem.add_self_loop_penalty(lambda_l=lambda_l)
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
            return dfa
    raise RuntimeError("DFA with at most {} states not found".format(max_size))
