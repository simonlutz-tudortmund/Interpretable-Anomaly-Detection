import pandas as pd
import ast
import re

from src.fit_dfa_wrapers import fit_dfa
from src.milp.milp_data_file import learn_dfa


def convert_to_tuple(features_str):
    # Extract the elements using regex
    # This pattern finds all occurrences of E followed by numbers
    elements = re.findall(r"E\d+", features_str)
    return tuple(elements)


# Apply the conversion

# Apply the conversion
df = pd.read_csv("./data/HFDS/HFDS_v1.csv", index_col=0)

df["Features"] = df["Features"].apply(convert_to_tuple)

df_pruned = df[~df["Features"].duplicated()]

lower_bound = 0.2
upper_bound = 0.3
sample = df_pruned["Features"].values.tolist()
alphabet = frozenset(value for trace in sample for value in trace)

dfa = learn_dfa(
    sample=sample,
    alphabet=alphabet,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    min_dfa_size=3,
    lambda_l=0.5,
    lambda_s=0.0,
    lambda_p=0.5,
    verbose=2,
    pointwise=False,
)


print(dfa)
