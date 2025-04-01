from fit_dfa import fit_minimal_dfa


sample = [
    ("a", "b", "b"),
    ("a", "a"),
    ("a", "b"),
    ("b", "a"),
    ("a", "a", "b"),
    ("b", "a", "b"),
    ("a", "a", "a"),
    ("c", "c", "c", "c", "c"),
    ("c", "c", "c", "c", "c", "c"),
]
alphabet = {"a", "b", "c"}
lower_bound = 2
upper_bound = None

dfa = fit_minimal_dfa(
    sample=sample,
    alphabet=alphabet,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    min_dfa_size=2,
    lambda_l=0.5,
    lambda_s=0.5,
    lambda_p=0.5,
    verbose=2,
    pointwise=False,
)

if dfa:
    print("Minimal DFA found:")
    print(f"States: {dfa['states']}")
    print(f"Alphabet: {dfa['alphabet']}")
    print(f"Transitions: {dfa['transitions']}")
    print(f"Initial State: {dfa['initial_state']}")
    print(f"Final States: {dfa['final_states']}")
else:
    print("No feasible DFA found within the given bounds.")
