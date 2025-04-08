import argparse
from src.utility import (
    load_sample,
    evaluate_sample_with_dfa,
    calculate_metrics,
    log_metrics,
    add_common_args,
)
import logging
from typing import List, Tuple, Optional, Set
from src.fit_dfa_wrapers import fit_and_log_dfa


def train_and_evaluate_dfa(
    train_filepath: str,
    test_filepath: str,
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
):
    """
    Trains a DFA on the training dataset and evaluates its performance on a labeled test dataset.

    Parameters:
    - train_filepath: Path to the training sample file (unlabeled).
    - test_filepath: Path to the test sample file (labeled).
    - algorithm_number: Algorithm choice (1, 2, or 3).
    - distance_function: Distance function name (for algorithm 3).
    - lower_bound, upper_bound: Constraints for the DFA size.
    - min_dfa_size: Minimum size of the DFA.
    - is_numeric_data: Whether the input contains numeric data.
    - verbose: Logging verbosity level.

    Logs the evaluation results, including accuracy.
    """
    # Load train and test samples
    train_sample, alphabet = load_sample(train_filepath, is_numeric_data)
    test_sample, _, test_labels = load_sample(
        test_filepath, is_numeric_data, is_labeled=True
    )

    if train_sample is None or test_sample is None or test_labels is None:
        return

    # Train DFA
    dfa = fit_and_log_dfa(
        train_sample,
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
        output_path,
    )

    accepted, rejected = evaluate_sample_with_dfa(test_sample, dfa)

    # Convert accepted/rejected into a dictionary for easier comparison
    accepted_set = set(accepted)
    predicted_labels = [
        "accept" if word in accepted_set else "reject" for word in test_sample
    ]

    metrics = calculate_metrics(test_labels, predicted_labels)

    logging.info(f"Evaluation results:")
    log_metrics(metrics)
    logging.debug(f"Accepted words: {accepted}")
    logging.debug(f"Rejected words: {rejected}")


def cli():
    parser = argparse.ArgumentParser(description="Learn a DFA from a sample.")
    parser.add_argument(
        "--train_filepath",
        type=str,
        default="train2.txt",
        help="Path to the train sample file.",
    )
    parser.add_argument(
        "--test_filepath",
        type=str,
        default="train2.txt",
        help="Path to the test sample file.",
    )
    add_common_args(parser)

    args = parser.parse_args()

    log_levels = {
        0: logging.CRITICAL,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
    }

    logging.basicConfig(level=log_levels[args.verbose])

    train_and_evaluate_dfa(
        args.train_filepath,
        args.test_filepath,
        args.algorithm,
        args.distance,
        args.lower_bound,
        args.upper_bound,
        args.lambda_s,
        args.lambda_l,
        args.lambda_p,
        args.min_dfa_size,
        args.numeric_data,
        args.verbose,
        args.visualize,
        args.output_path,
    )


if __name__ == "__main__":
    cli()
