import argparse
from fit_dfa_wrapers import fit_dfa_from_file
import logging
from utility import add_common_args


def cli():
    parser = argparse.ArgumentParser(description="Learn a DFA from a sample.")
    parser.add_argument("--filepath", type=str, default="train2.txt", help="Path to the sample file.")
    add_common_args(parser)

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
                      args.lambda_s,
                      args.lambda_l,
                      args.lambda_p,
                      args.min_dfa_size,
                      args.numeric_data,
                      args.verbose,
                      args.visualize,
                      args.output_path)


if __name__ == "__main__":
    cli()
