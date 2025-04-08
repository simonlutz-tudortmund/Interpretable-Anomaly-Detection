from src.milp.milp_data_file import learn_dfa


def test_lower_and_upper_bound():
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
    upper_bound = 3

    dfa = learn_dfa(
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

    assert ("c", "c", "c", "c", "c") in dfa
    assert ("c", "c", "c", "c", "c", "c") in dfa
    if dfa:
        print("Minimal DFA found:")
        print(f"States: {dfa.states}")
        print(f"Alphabet: {dfa.alphabet}")
        print(f"Transitions: {dfa.transitions}")
        print(f"Initial State: {dfa.initial_state}")
        print(f"Final States: {dfa.final_states}")
    else:
        print("No feasible DFA found within the given bounds.")
