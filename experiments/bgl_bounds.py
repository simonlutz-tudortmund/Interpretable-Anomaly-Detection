from math import ceil, floor
import pandas as pd
import re

from src.milp.milp_data_file import learn_dfa_with_bounds, learn_dfa_with_labels
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)


def convert_to_tuple(features_str):
    # Extract the elements using regex
    # This pattern finds all occurrences of E followed by numbers
    elements = re.findall(r"E\d+", features_str)
    return tuple(elements)


# Apply the conversion

# Apply the conversion
df = pd.read_csv("./data/BGL/BGL.csv", index_col=0)
# feature_lengths = df["Features"].apply(len)

# # Plot the histogram with logarithmically spaced bins but a linear x-axis
# plt.figure(figsize=(10, 6))
# bins = np.logspace(
#     0.1, np.log10(feature_lengths.max()), 20
# )  # Logarithmically spaced bins
# plt.hist(feature_lengths, bins=bins, edgecolor="black")
# plt.xlabel("Length of Features")
# plt.ylabel("Frequency")
# plt.title("Histogram of Feature Lengths with Logarithmic Bins")
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.show()
df["Features"] = df["Features"].apply(convert_to_tuple)

# 75% of data
# df_pruned = df[~df["Features"].duplicated() & (df["Features"].apply(len) <= 20)]
df_pruned = df[df["Features"].apply(len) <= 30]

lower_bound = 0.159
upper_bound = 0.16
sample = df_pruned["Features"].values.tolist()
alphabet = frozenset(value for trace in sample for value in trace)
positive_sample = df_pruned[df_pruned["Label"] == "Fail"]["Features"].values.tolist()
negative_sample = df_pruned[df_pruned["Label"] == "Success"]["Features"].values.tolist()


dfa_unlabeled, _ = learn_dfa_with_bounds(
    alphabet=alphabet,
    sample=sample,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    min_dfa_size=4,
    lambda_l=None,
    lambda_s=None,
    lambda_p=None,
    verbose=2,
)
df_pruned["Prediction"] = (
    df_pruned["Features"].apply(lambda feature: feature in dfa_unlabeled).astype(int)
)
df_pruned["Label_Binary"] = df_pruned["Label"].map({"Success": 0, "Fail": 1})

# Compute the confusion matrix
labels = df_pruned["Label_Binary"]
predictions = df_pruned["Prediction"]  # Ensure predictions are integers

# Compute precision, recall, F1-score, and other metrics
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Alternatively, print a full classification report
print("\nClassification Report:")
print(classification_report(labels, predictions, target_names=["Fail", "Success"]))

print(dfa_unlabeled)

from src.utils.dfa import CausalDFA
from graphviz import Digraph


def save_visualized_dfa(dfa: CausalDFA, output_path="dfa"):
    """
    Generates a visualization of a DFA and saves it as an image.

    Parameters:
    - dfa (dict): A dictionary representing the DFA with keys:
        'states' (iterable): The set of states.
        'alphabet' (set): The set of input symbols.
        'transitions' (dict): A mapping from (state, symbol) to next state.
        'initial_state' (int/str): The initial state.
        'final_states' (set): The set of accepting states.
    - output_path (str): Path to save the resulting image (without extension).
    """
    dot = Digraph(format="png")

    for state in dfa.states:
        shape = "doublecircle" if state in dfa.final_states else "circle"
        dot.node(str(state), shape=shape)

    dot.node("start", shape="none", width="0")
    dot.edge("start", str(dfa.initial_state))

    for state in dfa.states:
        for symbol, next_state in dfa.transitions[state].items():
            if state != next_state:
                dot.edge(str(state), str(next_state), label=str(symbol))

    dot.render(output_path, cleanup=True)


save_visualized_dfa(dfa_unlabeled)

"""
3 states
Precision: 0.12862040111210085
Recall: 0.1282361806340492
F1 Score: 0.128428003503807

Classification Report:
              precision    recall  f1-score   support

        Fail       0.83      0.84      0.83    195839
     Success       0.13      0.13      0.13     37158

    accuracy                           0.72    232997
   macro avg       0.48      0.48      0.48    232997
weighted avg       0.72      0.72      0.72    232997
"""

"""
4 states
Precision: 0.12851243015628797
Recall: 0.12812853221378975
F1 Score: 0.12832019405700423

Classification Report:
              precision    recall  f1-score   support

        Fail       0.83      0.84      0.83    195839
     Success       0.13      0.13      0.13     37158

    accuracy                           0.72    232997
   macro avg       0.48      0.48      0.48    232997
weighted avg       0.72      0.72      0.72    232997
"""
