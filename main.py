import argparse
from fit_dfa_wrapers import fit_dfa_from_file
import logging


def cli():
    parser = argparse.ArgumentParser(description="Learn a DFA from a sample.")
    parser.add_argument("--filepath", type=str, default="sample.txt", help="Path to the sample file.")
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

    fit_dfa_from_file(args.filepath,
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
