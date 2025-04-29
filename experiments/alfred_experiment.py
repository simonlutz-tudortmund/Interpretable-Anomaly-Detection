import csv
import logging
import os
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from experiments.data_loader import load_alfred_data
from milp.milp_data_file import learn_dfa_with_bounds
from milp.problem import Problem
from utils.dfa import CausalDFA
from utils.paths import get_experiments_path
from utils.util import get_bounds

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
goals = [0, 1, 2, 3, 4, 5, 6]
bound_deviations = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
os.makedirs("logs", exist_ok=True)
os.makedirs("dfa_visualizations", exist_ok=True)

# CSV Setup
fieldnames = [
    "goal",
    "deviation",
    "case_name",
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
base_folder = get_experiments_path().joinpath("alfred")
CSV_PATH = str(base_folder.joinpath("results.csv"))


# Initialize CSV with header
with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


def setup_logger(log_file):
    """Configure logger with specific file handler"""
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers = []
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    return logger


for goal in goals:
    train_df, test_df, alphabet = load_alfred_data(goal=goal)
    exact_bound = len(train_df[train_df["Label"] == 1]) / len(train_df)
    lower_starting, upper_starting = get_bounds(exact_bound, 2)

    for deviation in bound_deviations:
        lower = max(0.0, float(lower_starting) - deviation)
        upper = min(1.0, float(upper_starting) + deviation)

        boundary_cases = [
            (lower, upper, "both_bounds"),
            (lower, None, "lower_only"),
            (None, upper, "upper_only"),
        ]

        for lower_bound, upper_bound, case_name in boundary_cases:
            file_suffix = f"goal{goal}_dev{deviation}_{case_name}"
            log_file = str(base_folder.joinpath("logs", file_suffix))
            dfa_file = str(base_folder.joinpath("dfas", file_suffix))
            logger = setup_logger(log_file)
            row_data = {
                "goal": goal,
                "deviation": deviation,
                "case_name": case_name,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "exact_bound": exact_bound,
                "runtime": "N/A",
                "precision": "N/A",
                "recall": "N/A",
                "f1": "N/A",
                "states": "N/A",
                "status": "failed",
                "error_message": "",
            }

            try:
                # Learn DFA and calculate metrics
                dfa, problem = learn_dfa_with_bounds(
                    sample=train_df["Features"].values.tolist(),
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    alphabet=alphabet,
                    min_dfa_size=2,
                    lambda_l=None,
                    lambda_s=None,
                    lambda_p=None,
                    verbose=2,
                )

                # Save visualization
                dfa.save_visualized_dfa(output_path=dfa_file)

                # Generate predictions
                test_df["Prediction"] = test_df["Features"].apply(
                    lambda x: int(x in dfa)
                )

                # Update metrics
                metrics = {
                    "precision": precision_score(
                        test_df["Label"], test_df["Prediction"]
                    ),
                    "recall": recall_score(test_df["Label"], test_df["Prediction"]),
                    "f1": f1_score(test_df["Label"], test_df["Prediction"]),
                }

                # Update row data
                row_data.update(
                    {
                        "runtime": getattr(problem.model, "Runtime", "N/A"),
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "states": len(dfa.states),
                        "status": "success",
                    }
                )

                # Log detailed information
                logger.info(f"DFA learned successfully\n{dfa}")
                logger.info(f"Runtime: {row_data['runtime']}")
                logger.info(
                    classification_report(test_df["Label"], test_df["Prediction"])
                )

            except Exception as e:
                row_data["error_message"] = str(e)
                logger.error(f"Test failed: {e}", exc_info=True)

            finally:
                # Write to CSV
                with open(CSV_PATH, "a", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(row_data)

                # Cleanup logger
                for handler in logger.handlers:
                    handler.close()
                    logger.removeHandler(handler)


# Load the CSV data
df = pd.read_csv(CSV_PATH)

df["case_type"] = df["case_name"].map(
    {"both_bounds": "TB", "upper_only": "UB", "lower_only": "LB"}
)

# Set up plot
plt.figure(figsize=(16, 10))
sns.set_style(style="whitegrid")
color_palette = sns.color_palette("husl", n_colors=len(df["goal"].unique()))
line_styles = {"TB": "-", "UB": "--", "LB": ":"}
markers = {"TB": "o", "UB": "s", "LB": "D"}

# Plot each combination
for goal in [0, 2, 5]:
    goal_df = df[df["goal"] == goal]
    for case in ["TB", "UB", "LB"]:
        case_df = goal_df[goal_df["case_type"] == case].sort_values("deviation")
        if not case_df.empty:
            plt.plot(
                case_df["deviation"],
                case_df["f1"],
                color=color_palette[goal],
                linestyle=line_styles[case],
                marker=markers[case],
                markersize=8,
                linewidth=2.5,
                label=f"Goal {goal} {case}",
            )

# Configure plot aesthetics
plt.title("F1 Score vs Acceptance Bound Deviation", pad=20)
plt.xlabel("Acceptance Bound Deviation", labelpad=15)
plt.ylabel("F1 Score", labelpad=15)
plt.ylim(0, 1.05)
plt.xlim(left=0)

# Create combined legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(
    handles,
    labels,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0.0,
    frameon=True,
    title="Goal & Bound Type",
)

plt.tight_layout()
plt.savefig("all_goals_f1_vs_deviation.png")
plt.close()
