import argparse
import sys

sys.path.append(".")
sys.path.append("./src")
import csv
import logging
import os
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from experiments.data_loader import load_hfds_data
from src.milp.milp_data_file import learn_dfa_with_bounds
from src.utils.paths import get_experiments_path
from src.utils.util import get_bounds

# Configuration
# CLASSES = range(7)  # Assuming classes 0-6
# class_pairs = list(permutations(CLASSES, 2))  # All ordered pairs
SEQUENCE_LENGTHS = [15]
TEST_SIZE = 0.2  # 20% for testing
BOUND_DEVIATIONS = [0.0]
INTERPRETABILITY_CASES = [
                    (None, None, None, "standart"),
                    (0.05, None,None, "self_loops"),
                    (None, 0.05, None, "sink_states"),
                    (None, None, 0.05, "parallel_edges"),
                    (0.05, 0.05, None, "self_loops_and_sink_states"),
                    (0.05, None, 0.05, "self_loops_and_parallel_edges"),
                    (None, 0.05, 0.05, "sink_states_and_parallel_edges"),
                    (0.05, 0.05, 0.05, "all"),
                ]


# Setup directories
base_folder = get_experiments_path().joinpath("hdfs").joinpath("interpretability")
os.makedirs(base_folder.joinpath("logs"), exist_ok=True)
os.makedirs(base_folder.joinpath("gurobi_logs"), exist_ok=True)
os.makedirs(base_folder.joinpath("dfas"), exist_ok=True)


# CSV Setup
CSV_PATH = base_folder.joinpath("results_bounds.csv")
parser = argparse.ArgumentParser()
parser.add_argument('--seeds', nargs='+', type=int, help='Seeds to process')
args, unknown = parser.parse_known_args()

if args.seeds:
    SEEDS = args.seeds
else:
    # CSV Setup
    SEEDS = [
        114,
        28998,
        7239,
        11517,
        79820,
        9471,
        36624,
        39871,
        56085,
        89095,
        98846,
        3075,
        39161,
        8988,
        70332,
        51338,
        8938,
        12153,
        72994,
        55151,
        9178,
        31055,
        97635,
        63198,
        85303,
        37843,
        11298,
        58656,
        87228,
        93884,
        58283,
        162,
        93344,
        32584,
        65447,
        84787,
        59934,
        6886,
        89149,
        26253,
        75385,
        88322,
        51134,
        68503,
        29382,
        45,
        69007,
        15852,
        79121,
        33653,
    ]
fieldnames = [
    "seed",
    "max_sequence_length",
    "case_name",
    "deviation",
    "lower_bound",
    "upper_bound",
    "exact_bound",
    "interpretability_case",
    "self_loop_penalty",
    "sink_state_penalty",
    "parallel_edge_penalty",
    "runtime",
    "precision",
    "recall",
    "f1",
    "states",
    "status",
    "error_message",
]

with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers = []
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


for max_sequence_length in SEQUENCE_LENGTHS:
    for seed in SEEDS:
        for (self_loop_penalty, sink_state_penalty, parallel_edge_penalty, interpretability_case) in INTERPRETABILITY_CASES:
            try:
                train_df, test_df, alphabet = load_hfds_data(
                    seed=seed,
                    test_percentage=TEST_SIZE,
                    max_sequence_length=max_sequence_length,
                )
                exact_bound = len(train_df[train_df.Label == 1]) / len(train_df)
                lower_start, upper_start = get_bounds(exact_bound, 3)

                lower = max(0.0, float(lower_start))
                upper = min(1.0, float(upper_start))

                boundary_cases = [
                    (lower, upper, "both_bounds"),
                    # (lower, None, "lower_only"),
                    # (None, upper, "upper_only"),
                ]

                for lb, ub, case_name in boundary_cases:
                    log_file = str(
                        base_folder.joinpath(
                            "logs",
                            f"{case_name}_{max_sequence_length}_{interpretability_case}_{seed}.log",
                        )
                    )
                    dfa_file = str(
                        base_folder.joinpath(
                            "dfas",
                            f"{case_name}_{max_sequence_length}_{interpretability_case}_{seed}",
                        )
                    )
                    log_file_gurobi = str(
                        base_folder.joinpath(
                            "gurobi_logs",
                            f"{case_name}_{max_sequence_length}_{interpretability_case}_{seed}_gurobi.log",
                        )
                    )
                    logger = setup_logger(log_file)

                    row_data = {
                        "seed": seed,
                        "max_sequence_length": max_sequence_length,
                        "interpretability_case": interpretability_case,
                        "self_loop_penalty": self_loop_penalty,
                        "sink_state_penalty": sink_state_penalty,
                        "parallel_edge_penalty": parallel_edge_penalty,
                        "case_name": case_name,
                        "lower_bound": lb,
                        "upper_bound": ub,
                        "exact_bound": exact_bound,
                        "status": "failed",
                        "error_message": "",
                    }

                    try:
                        dfa, problem = learn_dfa_with_bounds(
                            sample=train_df.Features.tolist(),
                            lower_bound=lb,
                            upper_bound=ub,
                            alphabet=alphabet,
                            min_dfa_size=3,
                            verbose=2,
                            lambda_l=self_loop_penalty,
                            lambda_s=sink_state_penalty,
                            lambda_p=parallel_edge_penalty,
                            log_file=log_file_gurobi,
                        )

                        test_df["Prediction"] = test_df.Features.apply(
                            lambda x: int(x in dfa)
                        )

                        row_data.update(
                            {
                                "runtime": getattr(problem.model, "Runtime", "N/A"),
                                "precision": precision_score(
                                    test_df.Label, test_df.Prediction, zero_division=0
                                ),
                                "recall": recall_score(
                                    test_df.Label, test_df.Prediction, zero_division=0
                                ),
                                "f1": f1_score(
                                    test_df.Label, test_df.Prediction, zero_division=0
                                ),
                                "states": len(dfa.states),
                                "status": "success",
                            }
                        )

                        dfa.save_visualized_dfa(dfa_file)
                        logger.info(
                            f"Learned DFA saved to {dfa_file}\n"
                            f"States: {len(dfa.states)}\n"
                            f"Initital State: {dfa.initial_state}\n"
                            f"Final States: {dfa.final_states}\n"
                            f"Alphabet: {dfa.alphabet}\n"
                            f"Transitions:\n"
                            f"{'\n'.join([f'{from_state} -> {symbol} -> {to_state}' for from_state, symbols in dfa.transitions.items() for symbol, to_state in symbols.items()])}"
                        )
                        logger.info(
                            f"\nSuccess\n{dfa}\n{classification_report(test_df.Label, test_df.Prediction)}"
                        )

                    except Exception as e:
                        row_data["error_message"] = str(e)
                        logger.error(f"Error: {str(e)}", exc_info=True)

                    finally:
                        with open(CSV_PATH, "a") as csvfile:
                            csv.DictWriter(csvfile, fieldnames=fieldnames).writerow(
                                row_data
                            )
                        logger.handlers.clear()

            except Exception as e:
                print(f"Error processing seed ({seed}): {str(e)}")
