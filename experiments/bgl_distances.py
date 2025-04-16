import pandas as pd
import re

from src.milp.milp_data_file import learn_dfa_with_distances_quadratic
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

df = pd.read_csv("./data/BGL/BGL.csv", index_col=0)
df["Features"] = df["Features"].apply(convert_to_tuple)

# 75% of data
# df_pruned = df[~df["Features"].duplicated() & (df["Features"].apply(len) <= 20)]
df_pruned = df[df["Features"].apply(len) <= 15]

sample = df_pruned["Features"].values.tolist()
alphabet = frozenset(value for trace in sample for value in trace)


dfa_unlabeled = learn_dfa_with_distances_quadratic(
    alphabet=alphabet,
    sample=sample,
    distance_function="levenshtein",
    dfa_size=4,
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
Precision: 0.06638643488173268
Recall: 0.7340475815345833
F1 Score: 0.1217609471167039

Classification Report:
              precision    recall  f1-score   support

        Fail       0.96      0.36      0.52    101890
     Success       0.07      0.73      0.12      6347

    accuracy                           0.38    108237
   macro avg       0.51      0.55      0.32    108237
weighted avg       0.90      0.38      0.50    108237
"""
