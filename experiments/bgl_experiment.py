import sys

sys.path.append(".")
sys.path.append("./src")
import csv
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from experiments.data_loader import load_bgl_data, load_hfds_data
from src.milp.milp_data_file import learn_dfa_with_bounds
from src.utils.paths import get_experiments_path
from src.utils.util import get_bounds

# Configuration
# CLASSES = range(7)  # Assuming classes 0-6
# class_pairs = list(permutations(CLASSES, 2))  # All ordered pairs
SEQUENCE_LENGTHS = [25]
TEST_SIZE = 0.2  # 20% for testing
BOUND_DEVIATIONS = [0.0]

# Setup directories
base_folder = get_experiments_path().joinpath("bgl")
os.makedirs(base_folder.joinpath("logs"), exist_ok=True)
os.makedirs(base_folder.joinpath("dfas"), exist_ok=True)

# CSV Setup
CSV_PATH = base_folder.joinpath("results_full.csv")
SEEDS = [
    # 114,
    28998,
    7239,
    11517,
    79820,
    9471,
    36624,
    39871,
    56085,
    89095,
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
            train_df, test_df, alphabet = load_bgl_data(
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


# Visualization
df = pd.read_csv(CSV_PATH)
df["case_type"] = df.case_name.map(
    {"both_bounds": "TB", "lower_only": "LB", "upper_only": "UB"}
)

# Visualization
agg_df = (
    df.groupby(["case_type", "deviation"])
    .agg(mean_f1=("f1", "mean"), std_f1=("f1", "std"))
    .reset_index()
)

plt.figure(figsize=(16, 10))
sns.set_style(style="whitegrid")
color_palette = sns.color_palette("husl", n_colors=3)
color_palette = {
    "TB": color_palette[0],
    "UB": color_palette[1],
    "LB": color_palette[2],
}
line_styles = {"TB": "-", "UB": "--", "LB": ":"}
markers = {"TB": "o", "UB": "s", "LB": "D"}

# Create plot
for case_type in ["TB", "UB", "LB"]:
    case_data = agg_df[agg_df.case_type == case_type].sort_values("deviation")
    deviations = case_data.deviation
    means = case_data.mean_f1
    stds = case_data.std_f1

    plt.plot(
        deviations,
        means,
        label=case_type,
        color=color_palette[case_type],
        linestyle=line_styles[case_type],
        marker=markers[case_type],
        markersize=8,
        linewidth=2.5,
    )

    plt.fill_between(
        deviations,
        means - stds,
        means + stds,
        color=color_palette[case_type],
        alpha=0.2,
    )

plt.title("Average F1 Score vs Acceptance Bound Deviation", pad=15)
plt.xlabel("Bound Deviation", labelpad=10)
plt.ylabel("F1 Score", labelpad=10)
plt.ylim(0, 1.05)
plt.xlim(-0.001, max(BOUND_DEVIATIONS) + 0.001)
plt.grid(True, alpha=0.3)

# Create legend
handles, labels = plt.gca().get_legend_handles_labels()
new_handles = [
    plt.Line2D(
        [0], [0], color=color_palette[label], linestyle=line_styles[label], lw=2.5
    )
    for label in labels
]
plt.legend(
    new_handles,
    labels,
    title="Bound Type",
    loc="lower left" if agg_df.mean_f1.min() < 0.5 else "upper right",
    frameon=True,
    framealpha=0.9,
)

plt.tight_layout()
plt.savefig("aggregated_analysis_permutated.png", dpi=300)
plt.show()
