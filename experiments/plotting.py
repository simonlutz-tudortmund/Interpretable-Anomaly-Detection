import matplotlib as mpl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.paths import get_experiments_path


def plot_f1_score_distribution():
    # List of your CSV files
    base_folder = get_experiments_path()

    # Mapping of file names to dataset names
    csv_files = {
        "ALFRED": str(base_folder.joinpath("alfred", "results_bounds.csv")),
        "HDFS": str(base_folder.joinpath("hdfs", "results_bounds.csv")),
        "BGL": str(base_folder.joinpath("bgl", "results_bounds.csv")),
    }
    # Mapping of case_name to algorithm names
    algorithm_mapping = {
        "both_bounds": "Two-Bound",
        "upper_only": "Distance-Based",
        "lower_only": "Single-Bound",
    }

    # List to hold individual DataFrames
    data_frames = []

    # Read and process each CSV file
    for dataset_name, file_name in csv_files.items():
        df = pd.read_csv(file_name)
        df = df[df["deviation"] == 0]  # Filter for deviation = 0
        df = df[
            ["case_name", "f1"]
        ].dropna()  # Extract relevant columns and drop missing values
        df["dataset"] = dataset_name  # Add dataset name
        df["algorithm"] = df["case_name"].map(
            algorithm_mapping
        )  # Map to algorithm names
        data_frames.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Set the visual style
    sns.set_style(style="whitegrid")

    # Create the boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="algorithm", y="f1", hue="dataset", data=combined_df, palette="Set2")

    # Add titles and labels
    # plt.title("F1 Score Distribution by Algorithm Across Datasets")
    plt.ylabel("F1 Score")
    plt.xlabel("")

    # plt.legend(title="Dataset")

    # Create legend
    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(
        handles,
        ["ALFRED", "HDFS", "BGL"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.175),  # Position the legend above the plot
        frameon=True,
        framealpha=0.9,
        ncol=3,  # Arrange legend items in a single row
    )

    # Display the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        str(base_folder.joinpath("plots", "f1_score_distribution.png")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        str(base_folder.joinpath("plots", "f1_score_distribution.pdf")),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_runtime_distribution():
    base_folder = get_experiments_path()

    # Mapping of file names to dataset names
    csv_files = {
        "ALFRED": str(base_folder.joinpath("alfred", "results_bounds.csv")),
        "HDFS": str(base_folder.joinpath("hdfs", "results_bounds.csv")),
        "BGL": str(base_folder.joinpath("bgl", "results_bounds.csv")),
    }
    # Mapping of case_name to algorithm names
    algorithm_mapping = {
        "both_bounds": "Two-Bound",
        "upper_only": "Distance-Based",
        "lower_only": "Single-Bound",
    }

    # List to hold individual DataFrames
    data_frames = []

    # Read and process each CSV file
    for dataset_name, file_name in csv_files.items():
        df = pd.read_csv(file_name)
        df = df[df["deviation"] == 0]  # Filter for deviation = 0
        df = df[
            ["case_name", "runtime"]
        ].dropna()  # Extract relevant columns and drop missing values
        df["dataset"] = dataset_name  # Add dataset name
        df["algorithm"] = df["case_name"].map(
            algorithm_mapping
        )  # Map to algorithm names
        data_frames.append(df)

    # Combine all DataFrames into one
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Set the visual style
    sns.set_style(style="whitegrid")

    # Create the boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        x="algorithm", y="runtime", hue="dataset", data=combined_df, palette="Set2"
    )

    # Add titles and labels
    # plt.title("F1 Score Distribution by Algorithm Across Datasets")
    plt.ylabel("Runtime (seconds)")
    plt.xlabel("")

    # Create legend
    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(
        handles,
        ["ALFRED", "HDFS", "BGL"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.175),  # Position the legend above the plot
        frameon=True,
        framealpha=0.9,
        ncol=3,  # Arrange legend items in a single row
    )

    # Display the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        str(
            base_folder.joinpath("plots", "runtime_distribution.png"),
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        str(
            base_folder.joinpath("plots", "runtime_distribution.pdf"),
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_bound_deviation():
    base_folder = get_experiments_path()
    # CSV Setup
    csv_path = base_folder.joinpath("alfred", "results_bounds.csv")
    bound_deviations = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    # Visualization
    df = pd.read_csv(csv_path)
    df["case_type"] = df.case_name.map(
        {"both_bounds": "Two-Bound", "upper_only": "Single-Bound"}
    )

    # Aggregate results across all class pairs
    agg_df = (
        df.groupby(["case_type", "deviation"])
        .agg(mean_f1=("f1", "mean"), std_f1=("f1", "std"))
        .reset_index()
    )

    plt.figure(figsize=(6, 3))
    sns.set_style(style="whitegrid")
    color_palette = sns.color_palette("husl", n_colors=3)
    color_palette = {
        "Two-Bound": color_palette[0],
        "Single-Bound": color_palette[1],
    }
    line_styles = {"Two-Bound": "-", "Single-Bound": "--"}
    markers = {"Two-Bound": "o", "Single-Bound": "s"}

    # Create plot
    for case_type in ["Two-Bound", "Single-Bound"]:
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

    # plt.title("Average F1 Score vs Acceptance Bound Deviation", pad=15)
    plt.xlabel("Bound Deviation ±", labelpad=10)
    plt.ylabel("F1 Score", labelpad=10)
    plt.ylim(0, 1.175)
    plt.xlim(-0.005, max(bound_deviations) + 0.005)
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
        # title="Bound Type",
        loc="upper center",
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 1.30),  # Position the legend above the plot
        ncol=3,
    )
    plt.tight_layout()

    plt.savefig(
        str(
            base_folder.joinpath("plots", "alfred_bound_deviations.png"),
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        str(
            base_folder.joinpath("plots", "alfred_bound_deviations.pdf"),
        ),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        # Specify the serif font to match LNCS requirements
        "font.serif": ["Computer Modern Roman"],  # or ["Times"] if using Times
        "axes.labelsize": 20,
        "font.size": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
)
plot_runtime_distribution()
plot_f1_score_distribution()
mpl.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        # Specify the serif font to match LNCS requirements
        "font.serif": ["Computer Modern Roman"],  # or ["Times"] if using Times
        "axes.labelsize": 12.5,
        "font.size": 12.5,
        "legend.fontsize": 12.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)
plot_bound_deviation()
