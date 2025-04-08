# from fit_dfa import fit_minimal_dfa


# sample = [
#     ("a", "b", "b"),
#     ("a", "a"),
#     ("a", "b"),
#     ("b", "a"),
#     ("a", "a", "b"),
#     ("b", "a", "b"),
#     ("a", "a", "a"),
#     ("c", "c", "c", "c", "c"),
#     ("c", "c", "c", "c", "c", "c"),
# ]
# alphabet = {"a", "b", "c"}
# lower_bound = 2
# upper_bound = None

# dfa = fit_minimal_dfa(
#     sample=sample,
#     alphabet=alphabet,
#     lower_bound=lower_bound,
#     upper_bound=upper_bound,
#     min_dfa_size=2,
#     lambda_l=0.5,
#     lambda_s=0.0,
#     lambda_p=0.5,
#     verbose=2,
#     pointwise=False,
# )

# if dfa:
#     print("Minimal DFA found:")
#     print(f"States: {dfa['states']}")
#     print(f"Alphabet: {dfa['alphabet']}")
#     print(f"Transitions: {dfa['transitions']}")
#     print(f"Initial State: {dfa['initial_state']}")
#     print(f"Final States: {dfa['final_states']}")
# else:
#     print("No feasible DFA found within the given bounds.")

import pandas as pd
import ast
import re

from fit_dfa_wrapers import fit_dfa


def convert_to_tuple(features_str):
    # Extract the elements using regex
    # This pattern finds all occurrences of E followed by numbers
    elements = re.findall(r"E\d+", features_str)
    return tuple(elements)


# Apply the conversion

# Apply the conversion
df = pd.read_csv("./HFDS/HFDS_v1.csv", index_col=0)

df["Features"] = df["Features"].apply(convert_to_tuple)

df_pruned = df[~df["Features"].duplicated()]

lower_bound = 0.2
upper_bound = 0.3
sample = df_pruned["Features"].values.tolist()
alphabet = set(value for trace in sample for value in trace)

dfa = fit_dfa(
    algorithm_number="1",
    sample=sample,
    alphabet=alphabet,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    min_dfa_size=2,
    lambda_l=0.5,
    lambda_s=0.0,
    lambda_p=0.5,
    verbose=2,
    pointwise=False,
)
