from distance_functions import distance_function_names
from utility import load_sample, evaluate_sample_with_dfa, save_visualized_dfa, log_dfa_info
import logging
from typing import List, Tuple, Optional, Set
from fit_dfa import fit_minimal_dfa, fit_distance_based_dfa


def fit_dfa(sample: List[Tuple[str, ...]], alphabet: Set[str], algorithm_number: str,
            lower_bound: int, upper_bound: int, lambda_s: float, lambda_l: float, lambda_p: float,
            min_dfa_size: int, distance_function=None, verbose=0):
    """Trains a DFA using the specified algorithm and returns the learned DFA."""
    if algorithm_number in {"1", "2"}:
        return fit_minimal_dfa(sample, alphabet, lower_bound, upper_bound, lambda_s, lambda_l, lambda_p, min_dfa_size, verbose=verbose)
    elif algorithm_number == "3":
        return fit_distance_based_dfa(sample, alphabet, min_dfa_size, distance_function, lambda_s, lambda_l, lambda_p, verbose=verbose)
    return None


def fit_and_log_dfa(sample: List[Tuple[str, ...]],
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
                    output_path: Optional[str] = None):
    """
    Fits a DFA using the specified algorithm and parameters.
    If visualization is enabled, saves the DFA visualization to the given path.
    """
    distance_func = distance_function_names.get(distance_function)
    dfa = fit_dfa(sample, alphabet, algorithm_number, lower_bound, upper_bound, lambda_s, lambda_l, lambda_p, min_dfa_size, distance_func, verbose)

    if dfa:
        log_dfa_info(dfa)
    else:
        logging.info("No feasible DFA found within the given bounds.")

    if visualize and output_path:
        save_visualized_dfa(dfa, output_path)

    if verbose >= 3:
        accepted, rejected = evaluate_sample_with_dfa(sample, dfa)
        logging.debug(f"Accepted words: {accepted}")
        logging.debug(f"Rejected words: {rejected}")

    return dfa


def fit_dfa_from_file(sample_filepath: str,
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
                      output_path: Optional[str] = None):
    """
    Fits a DFA using the specified algorithm and parameters.
    If visualization is enabled, saves the DFA visualization to the given path.
    """
    sample, alphabet = load_sample(sample_filepath, is_numeric_data)
    if sample is None:
        return

    fit_and_log_dfa(sample,
                    alphabet,
                    algorithm_number,
                    distance_function,
                    lower_bound,
                    upper_bound,
                    lambda_s,
                    lambda_l,
                    lambda_p,
                    min_dfa_size,
                    verbose,
                    visualize,
                    output_path)
