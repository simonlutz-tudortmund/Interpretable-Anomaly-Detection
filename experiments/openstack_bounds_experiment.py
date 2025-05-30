import sys

from experiments.data_loader import load_openstack_data

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
from src.milp.milp_data_file import learn_dfa_with_bounds
from src.utils.paths import get_experiments_path
from src.utils.util import get_bounds

# Configuration
# CLASSES = range(7)  # Assuming classes 0-6
# class_pairs = list(permutations(CLASSES, 2))  # All ordered pairs
SEQUENCE_LENGTHS = [1000000]
TEST_SIZE = 0.2  # 20% for testing
BOUND_DEVIATIONS = [0.0]

# Setup directories
base_folder = get_experiments_path().joinpath("openstack")
os.makedirs(base_folder.joinpath("logs"), exist_ok=True)
os.makedirs(base_folder.joinpath("gurobi_logs"), exist_ok=True)
os.makedirs(base_folder.joinpath("dfas"), exist_ok=True)


# CSV Setup
CSV_PATH = base_folder.joinpath("results_distances.csv")
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
        try:
            train_df, test_df, alphabet = load_openstack_data(
                seed=seed,
                test_percentage=TEST_SIZE,
                max_sequence_length=max_sequence_length,
            )
            exact_bound = len(train_df[train_df.Label == 1]) / len(train_df)
            lower_start, upper_start = get_bounds(exact_bound, 3)

            for deviation in BOUND_DEVIATIONS:
                lower = max(0.0, float(lower_start) - deviation)
                upper = min(1.0, float(upper_start) + deviation)

                boundary_cases = [
                    (lower, upper, "both_bounds"),
                    (lower, None, "lower_only"),
                    (None, upper, "upper_only"),
                ]

                for lb, ub, case_name in boundary_cases:
                    log_file = str(
                        base_folder.joinpath(
                            "logs",
                            f"{case_name}_{max_sequence_length}_{deviation}_{seed}.log",
                        )
                    )
                    dfa_file = str(
                        base_folder.joinpath(
                            "dfas",
                            f"{case_name}_{max_sequence_length}_{deviation}_{seed}",
                        )
                    )
                    log_file_gurobi = str(
                        base_folder.joinpath(
                            "gurobi_logs",
                            f"{case_name}_{max_sequence_length}_{deviation}_{seed}_gurobi.log",
                        )
                    )
                    logger = setup_logger(log_file)

                    row_data = {
                        "seed": seed,
                        "max_sequence_length": max_sequence_length,
                        "deviation": deviation,
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
                            min_dfa_size=2,
                            verbose=2,
                            lambda_l=None,
                            lambda_s=None,
                            lambda_p=None,
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
