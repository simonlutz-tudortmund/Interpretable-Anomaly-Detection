import argparse
from utility import load_sample, evaluate_sample_with_dfa, calculate_metrics, log_metrics
import logging
from typing import List, Tuple, Optional, Set
from fit_dfa_wrapers import fit_and_log_dfa


def train_and_evaluate_dfa(train_filepath: str,
                           test_filepath: str,
                           algorithm_number: str,
                           distance_function: str,
                           lower_bound: int,
                           upper_bound: int,
                           min_dfa_size: int,
                           is_numeric_data: bool = False,
                           verbose: int = 0,
                           visualize: bool = False,
                           output_path: Optional[str] = None
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
    test_sample, _, test_labels = load_sample(test_filepath, is_numeric_data, is_labeled=True)

    if train_sample is None or test_sample is None or test_labels is None:
        return

    # Train DFA
    dfa = fit_and_log_dfa(train_sample,
                          alphabet,
                          algorithm_number,
                          distance_function,
                          lower_bound,
                          upper_bound,
                          min_dfa_size,
                          verbose,
                          visualize,
                          output_path)

    accepted, rejected = evaluate_sample_with_dfa(test_sample, dfa)

    # Convert accepted/rejected into a dictionary for easier comparison
    accepted_set = set(accepted)
    predicted_labels = ["accept" if word in accepted_set else "reject" for word in test_sample]

    metrics = calculate_metrics(test_labels, predicted_labels)

    logging.info(f"Evaluation results:")
    log_metrics(metrics)
    logging.debug(f"Accepted words: {accepted}")
    logging.debug(f"Rejected words: {rejected}")


def cli():
    parser = argparse.ArgumentParser(description="Learn a DFA from a sample.")
    parser.add_argument("--train_filepath", type=str, default="sample.txt",
                        help="Path to the train sample file.")
    parser.add_argument("--test_filepath", type=str, default="sample.txt",
                        help="Path to the test sample file.")
    parser.add_argument("--algorithm", type=str, choices=["1", "2", "3"], help="Algorithm to use.")
    parser.add_argument("--distance", type=str,
                        choices=["levenshtein", "jaccard", "hamming", "ordinal_hamming"], default="levenshtein",
                        help="Distance function to use.")
    parser.add_argument("--outlier_weight", type=float, default="0.5",
                        help="Weights given to outliers in contrast to regular objects "
                             "(used in distance based approach).")
    parser.add_argument("--lower_bound", type=int, help="Lower bound for the number of accepted words.")
    parser.add_argument("--upper_bound", type=int, help="Upper bound for the number of accepted words.")
    parser.add_argument("--min_dfa_size", type=int, default=1, help="Minimum DFA size.")
    parser.add_argument("--numeric_data", action="store_true",
                        help="Enable numeric data processing mode.")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize the DFA.")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=1,
                        help="Display and save visualization."
                             " 0 - only critical"
                             " 1 - show warnings"
                             " 2 - show info"
                             " 3 - show debug information")
    parser.add_argument("--output_path", type=str, default="./dfa.png", help="Path to save the visualization.")

    args = parser.parse_args()

    log_levels = {
        0: logging.CRITICAL,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG
    }

    logging.basicConfig(level=log_levels[args.verbose])

    train_and_evaluate_dfa(args.train_filepath,
                           args.test_filepath,
                           args.algorithm,
                           args.distance,
                           args.lower_bound,
                           args.upper_bound,
                           args.min_dfa_size,
                           args.numeric_data,
                           args.verbose,
                           args.visualize,
                           args.output_path)


if __name__ == "__main__":
    cli()