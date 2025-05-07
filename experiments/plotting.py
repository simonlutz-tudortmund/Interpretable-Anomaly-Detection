import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils.paths import get_experiments_path

# List of your CSV files
base_folder = get_experiments_path()

# Mapping of file names to dataset names
csv_files = {
    "ALFRED": str(base_folder.joinpath("alfred", "results_full.csv")),
    "HDFS": str(base_folder.joinpath("hdfs", "results_full.csv")),
    "BGL": str(base_folder.joinpath("bgl", "results_full.csv")),
}
# Mapping of case_name to algorithm names
algorithm_mapping = {
    "both_bounds": "Two Bounds",
    "upper_only": "Upper Bound",
    "lower_only": "Lower Bound",
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
    df["algorithm"] = df["case_name"].map(algorithm_mapping)  # Map to algorithm names
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
    bbox_to_anchor=(0.5, 1.05),  # Position the legend above the plot
    frameon=True,
    framealpha=0.9,
    ncol=3,  # Arrange legend items in a single row
)


# Display the plot
plt.tight_layout()
# plt.show()
plt.savefig("f1_score_distribution.png", dpi=300)
plt.savefig("f1_score_distribution.pdf", dpi=300)


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
    df["algorithm"] = df["case_name"].map(algorithm_mapping)  # Map to algorithm names
    data_frames.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(data_frames, ignore_index=True)

# Set the visual style
sns.set_style(style="whitegrid")

# Create the boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x="algorithm", y="runtime", hue="dataset", data=combined_df, palette="Set2")

# Add titles and labels
# plt.title("F1 Score Distribution by Algorithm Across Datasets")
plt.yscale("log")
plt.ylabel("Runtime (seconds, log scale)")
plt.xlabel("")

# Create legend
handles, labels = plt.gca().get_legend_handles_labels()

plt.legend(
    handles,
    ["ALFRED", "HDFS", "BGL"],
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),  # Position the legend above the plot
    frameon=True,
    framealpha=0.9,
    ncol=3,  # Arrange legend items in a single row
)


# Display the plot
plt.tight_layout()
# plt.show()
plt.savefig("runtime_distribution.png", dpi=300)
plt.savefig("runtime_distribution.pdf", dpi=300)
