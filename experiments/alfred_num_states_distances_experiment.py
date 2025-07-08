import argparse
import sys

sys.path.append(".")
sys.path.append("./src")
import csv
import logging
import math
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from itertools import permutations
from src.milp.milp_data_file import (
    learn_dfa_with_distances_quadratic,
)
from src.utils.paths import get_data_path, get_experiments_path

# Configuration
CLASSES = range(7)  # Assuming classes 0-6
class_pairs = list(permutations(CLASSES, 2))  # All ordered pairs
NUM_STATES = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example range for number of states
TEST_SIZE = 0.2  # 20% for testing

# Setup directories
base_folder = get_experiments_path().joinpath("alfred").joinpath("num_states")
os.makedirs(base_folder.joinpath("logs"), exist_ok=True)
os.makedirs(base_folder.joinpath("gurobi_logs"), exist_ok=True)
os.makedirs(base_folder.joinpath("dfas"), exist_ok=True)


CSV_PATH = base_folder.joinpath("results_distances.csv")
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
    "class_a",
    "class_b",   
    "num_states",
    "deviation",
    "case_name",
    "distance_function",
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


def setup_logger(log_file: str):
    """
    Setup logging configuration for the algorithm.

    Args:
        log_file (Optional[str]): Path to the log file for general logging.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )
    gurobi_logger = logging.getLogger("gurobipy")
    file_hdl = logging.FileHandler(log_file, mode="a")
    file_hdl.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    gurobi_logger.addHandler(file_hdl)

def load_data(class_a, class_b, seed=42):
    data = []
    with open(get_data_path().joinpath("Alfred", "alfred_full.txt"), "r") as f:
        for line in f:
            seq, label = line.strip().split(";")
            data.append({"Features": seq.split(","), "Label": int(label)})

    df = pd.DataFrame(data)
    df_a = df[df.Label == class_a]
    df_b = df[df.Label == class_b]

    a_train, a_test = train_test_split(df_a, test_size=TEST_SIZE, random_state=seed)
    # Split class_b into train (10%) and test (90%)
    b_train, b_test = train_test_split(df_b, test_size=TEST_SIZE, random_state=seed)

    # Use all of class_a for both train and test (caution: data leakage)
    train_df = pd.concat(
        [
            a_train.assign(Label=0),
            b_train.iloc[: math.ceil(len(a_train) / 9)].assign(Label=1),
        ]
    )
    test_df = pd.concat(
        [
            a_test.assign(Label=0),
            b_test.iloc[: math.ceil(len(a_test) / 9)].assign(Label=1),
        ]
    )

    # Generate alphabet from all features
    alphabet = sorted({feature for seq in df.Features for feature in seq})

    return train_df, test_df, alphabet


for seed in SEEDS:
    for class_a, class_b in class_pairs:
        try:
            train_df, test_df, alphabet = load_data(class_a, class_b, seed=seed)
            for num_states in NUM_STATES:  # Example range for number of states
                log_file = str(
                    base_folder.joinpath(
                        "logs",
                        f"levenstein_{seed}.log",
                    )
                )
                dfa_file = str(
                    base_folder.joinpath(
                        "dfas",
                        f"levenstein_{seed}",
                    )
                )
                log_file_gurobi = str(
                    base_folder.joinpath(
                        "gurobi_logs",
                        f"levenstein_gurobi_{seed}.log",
                    )
                )
                setup_logger(log_file)

                row_data = {
                    "seed": seed,
                    "class_a": class_a,
                    "class_b": class_b,
                    "num_states": num_states,
                    "case_name": "distance",
                    "distance_function": "levenshtein",
                    "status": "failed",
                    "error_message": "",
                }

                try:
                    dfa, problem = learn_dfa_with_distances_quadratic(
                        sample=train_df.Features.tolist(),
                        alphabet=alphabet,
                        dfa_size=num_states,
                        verbose=2,
                        lambda_l=None,
                        lambda_s=None,
                        lambda_p=None,
                        distance_function="levenshtein",
                        log_file=log_file_gurobi,
                    )

                    test_df["Prediction"] = test_df.Features.apply(lambda x: int(x in dfa))

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

                    logging.info(
                        f"Learned DFA saved to {dfa_file}\n"
                        f"States: {len(dfa.states)}\n"
                        f"Initital State: {dfa.initial_state}\n"
                        f"Final States: {dfa.final_states}\n"
                        f"Alphabet: {dfa.alphabet}\n"
                        f"Transitions:\n"
                        f"{'\n'.join([f'{from_state} -> {symbol} -> {to_state}' for from_state, symbols in dfa.transitions.items() for symbol, to_state in symbols.items()])}"
                    )
                    logging.info(
                        f"\nSuccess\n{dfa}\n{classification_report(test_df.Label, test_df.Prediction)}"
                    )

                except Exception as e:
                    row_data["error_message"] = str(e)
                    logging.error(f"Error: {str(e)}", exc_info=True)

                finally:
                    with open(CSV_PATH, "a") as csvfile:
                        csv.DictWriter(csvfile, fieldnames=fieldnames).writerow(row_data)
                        
        except Exception as e:
            print(f"Error processing pair ({class_a}, {class_b}, states {num_states}, seed {seed}): {str(e)}")
