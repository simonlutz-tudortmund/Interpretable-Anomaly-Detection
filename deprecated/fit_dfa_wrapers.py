from deprecated.distance_functions import distance_function_names
from deprecated.utility import (
    load_sample,
    evaluate_sample_with_dfa,
    save_visualized_dfa,
    log_dfa_info,
    get_prefixes,
)
import logging
from typing import List, Tuple, Optional, Set
from deprecated.fit_dfa import fit_minimal_dfa, fit_distance_based_dfa

from math import floor, ceil


def fit_dfa(
    sample: List[Tuple[str, ...]],
    alphabet: Set[str],
    algorithm_number: str,
    lower_bound: float,
    upper_bound: float,
    lambda_s: float,
    lambda_l: float,
    lambda_p: float,
    min_dfa_size: int,
    distance_function=None,
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

    if algorithm_number in {"1", "2"}:
        return fit_minimal_dfa(
            sample=sample,
            alphabet=alphabet,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            lambda_s=lambda_s,
            lambda_l=lambda_l,
            lambda_p=lambda_p,
            min_dfa_size=min_dfa_size,
            verbose=verbose,
            pointwise=pointwise,
        )
    elif algorithm_number == "3":
        return fit_distance_based_dfa(
            sample=sample,
            alphabet=alphabet,
            dfa_size=min_dfa_size,
            distance_function=distance_function,
            lambda_s=lambda_s,
            lambda_l=lambda_l,
            lambda_p=lambda_p,
            verbose=verbose,
            pointwise=pointwise,
        )
    return None


def fit_and_log_dfa(
    sample: List[Tuple[str, ...]],
    alphabet: Set[str],
    algorithm_number: str,
    distance_function: str,
    lower_bound: int,
    upper_bound: int,
    lambda_s: float,
    lambda_l: float,
    lambda_p: float,
    min_dfa_size: int,
    verbose: int = 0,
    visualize: bool = False,
    output_path: Optional[str] = None,
    pointwise=False,
):
    """
    Fits a DFA using the specified algorithm and parameters.
    If visualization is enabled, saves the DFA visualization to the given path.
    """
    distance_func = distance_function_names.get(distance_function)
    dfa = fit_dfa(
        sample=sample,
        alphabet=alphabet,
        algorithm_number=algorithm_number,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lambda_s=lambda_s,
        lambda_l=lambda_l,
        lambda_p=lambda_p,
        min_dfa_size=min_dfa_size,
        distance_function=distance_func,
        verbose=verbose,
        pointwise=pointwise,
    )

    if dfa:
        log_dfa_info(dfa=dfa)
    else:
        logging.info("No feasible DFA found within the given bounds.")

    if visualize and output_path:
        save_visualized_dfa(dfa=dfa, output_path=output_path)

    if verbose >= 3:
        if pointwise:
            prefixes = get_prefixes(sample=sample, start_token="")
            sample = prefixes
        accepted, rejected = evaluate_sample_with_dfa(sample=sample, dfa=dfa)
        logging.debug(f"Accepted words: {accepted}")
        logging.debug(f"Rejected words: {rejected}")

    return dfa


def fit_dfa_from_file(
    sample_filepath: str,
    algorithm_number: str,
    distance_function: str,
    lower_bound: int,
    upper_bound: int,
    lambda_s: float,
    lambda_l: float,
    lambda_p: float,
    min_dfa_size: int,
    is_numeric_data: bool = False,
    verbose: int = 0,
    visualize: bool = False,
    output_path: Optional[str] = None,
    pointwise=False,
):
    """
    Fits a DFA using the specified algorithm and parameters.
    If visualization is enabled, saves the DFA visualization to the given path.
    """
    sample, alphabet = load_sample(filepath=sample_filepath, is_numeric=is_numeric_data)
    if sample is None:
        return

    fit_and_log_dfa(
        sample=sample,
        alphabet=alphabet,
        algorithm_number=algorithm_number,
        distance_function=distance_function,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lambda_s=lambda_s,
        lambda_l=lambda_l,
        lambda_p=lambda_p,
        min_dfa_size=min_dfa_size,
        verbose=verbose,
        visualize=visualize,
        output_path=output_path,
        pointwise=pointwise,
    )
