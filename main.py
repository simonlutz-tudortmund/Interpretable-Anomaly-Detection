from gurobipy import Model, GRB
import numpy as np
from typing import List, Callable, Dict, Tuple
import editdistance


def algorithm_1(sample, alphabet, lower_bound, upper_bound):
    '''
    implementation of algorithm 1
    '''
    return learn_minimal_dfa(sample, alphabet, lower_bound, upper_bound)


def algorithm_2(sample, alphabet, lower_bound, upper_bound, min_dfa_size):
    '''
    implementation of algorithm 2. One of lower_bound and upper_bound can be None, min_dfa_size should be > 1
    :param min_dfa_size: algorithm finds dfa starting from size min_dfa_size
    '''
    return learn_minimal_dfa(sample, alphabet, lower_bound, upper_bound, min_dfa_size)


def get_prefixes(sample, start_token=""):
    prefixes = {(start_token,)} | {tuple(pref) for word in sample for pref in
                                   (tuple(word[:i + 1]) for i in range(len(word)))}
    return prefixes


def calculate_distance_matrix(
    sample: List[Tuple[str, ...]],
    distance_func: Callable[[Tuple[str, ...], Tuple[str, ...]], float]
) -> np.ndarray:
    """
    Calculate a distance matrix between all pairs of sequences in the sample.

    Parameters:
    - sample: List of sequences (each sequence is a tuple of strings).
    - distance_func: Function to calculate distance between two sequences.

    Returns:
    - A 2D numpy array where element (i, j) is the distance between sample[i] and sample[j].
    """
    n = len(sample)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = distance_func(sample[i], sample[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric
    return distance_matrix


def levenshtein_distance(seq1: Tuple[str, ...], seq2: Tuple[str, ...]) -> float:
    """Calculate the Levenshtein distance between two sequences of strings."""
    return editdistance.eval(seq1, seq2)


def jaccard_distance(seq1: Tuple[str, ...], seq2: Tuple[str, ...]) -> float:
    """
    Compute Jaccard distance between two sequences (as sets of elements).
    """
    set1, set2 = set(seq1), set(seq2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return 1 - intersection / union if union != 0 else 1.0


def find_max_dist(distance_matrix: np.ndarray) -> float:
    """Find the maximum distance in the distance matrix."""
    return np.max(distance_matrix)


def initialize_prefixes_and_bounds(sample, alphabet, lower_bound, upper_bound, min_dfa_size, start_token=""):
    """Initialize prefixes, bounds, and validate arguments."""
    if not lower_bound and not upper_bound:
        raise ValueError('Both lower_bound and upper_bound are not set')

    if lower_bound and upper_bound and lower_bound > upper_bound:
        raise ValueError('Invalid setup: lower_bound > upper_bound')

    if (not lower_bound or not upper_bound) and min_dfa_size == 1:
        raise ValueError('Invalid setup: one of lower_bound or upper_bound is not set while min_dfa_size == 1')

    start_token = ""
    prefixes = get_prefixes(sample, start_token)
    max_prefix_size = len(prefixes)
    max_dfa_size = max_prefix_size + 2
    return prefixes, max_dfa_size


def setup_model(n, alphabet, prefixes):
    """Set up the Gurobi model for DFA learning."""
    model = Model(f"DFA_size_{n}")
    states = range(n)

    delta = model.addVars(states, alphabet, states, vtype=GRB.BINARY, name="delta")
    final_states = model.addVars(states, vtype=GRB.BINARY, name="f")
    x = model.addVars(((w, q) for w in prefixes for q in states), vtype=GRB.BINARY, name="x")
    return model, states, delta, final_states, x


def add_transition_constraints(model, states, alphabet, delta):
    """Add constraints ensuring a single transition for each state-symbol pair."""
    for q in states:
        for a in alphabet:
            model.addConstr(sum(delta[q, a, q_prime] for q_prime in states) == 1, f"single_transition_{q}_{a}")


def add_prefix_constraints(model, states, prefixes, x):
    """Add constraints ensuring unique state for each prefix."""
    for w in prefixes:
        model.addConstr(sum(x[w, q] for q in states) == 1, f"single_run_{w}")


def add_initial_state_constraint(model, x, start_token):
    """Add a constraint specifying the initial state."""
    model.addConstr(x[(start_token,), 0] == 1, "initial_state")


def add_transition_follow_constraints(model, states, prefixes, alphabet, x, delta):
    """Add constraints for transition relationships between prefixes."""
    for w in prefixes:
        for q in states:
            for a in alphabet:
                wa = tuple(list(w) + [a])
                if wa in prefixes:
                    for q_prime in states:
                        model.addConstr(x[w, q] + delta[q, a, q_prime] - 1 <= x[wa, q_prime],
                                        f"run_{w}_{a}_{q}_{q_prime}_1")
                        model.addConstr(delta[q, a, q_prime] >= x[wa, q_prime], f"run_{w}_{a}_{q}_{q_prime}_2")


def add_alpha_constraints(model, states, sample, x, final_states):
    """Add constraints for prefix acceptance."""
    alpha = model.addVars(((s, q) for s in sample for q in states), vtype=GRB.BINARY, name="alpha")
    for w in sample:
        for q in states:
            model.addConstr(alpha[w, q] >= x[w, q] + final_states[q] - 1, f"accept_{w}_{q}_1")
            model.addConstr(alpha[w, q] <= x[w, q], f"accept_{w}_{q}_2")
            model.addConstr(alpha[w, q] <= final_states[q], f"accept_{w}_{q}_3")
    return alpha


def add_sample_constraints(model, states, sample, alpha, lower_bound, upper_bound):
    """Add constraints on the number of accepted words."""
    if lower_bound:
        model.addConstr(sum(alpha[w, q] for w in sample for q in states) >= lower_bound, "lower_bound")
    if upper_bound:
        model.addConstr(sum(alpha[w, q] for w in sample for q in states) <= upper_bound, "upper_bound")


def set_bounded_objective(model, states, sample, alpha, lower_bound, upper_bound):
    """Set the objective function for the model."""
    if lower_bound and upper_bound:
        model.setObjective(1, GRB.MINIMIZE)
    elif lower_bound:
        model.setObjective(sum(alpha[w, q] for w in sample for q in states), GRB.MINIMIZE)
    elif upper_bound:
        model.setObjective(sum(alpha[w, q] for w in sample for q in states), GRB.MAXIMIZE)


def extract_dfa_solution(model, states, alphabet, delta, final_states):
    """Extract the DFA solution from the model."""
    if model.status == GRB.OPTIMAL:
        dfa_transitions = {(q, a): q_prime for q in states for a in alphabet for q_prime in states
                           if delta[q, a, q_prime].x > 0.5}
        final_states_set = {q for q in states if final_states[q].x > 0.5}
        return {
            "states": states,
            "alphabet": alphabet,
            "transitions": dfa_transitions,
            "initial_state": 0,
            "final_states": final_states_set
        }
    return None


def set_dfa(sample, alphabet, dfa_size, prefixes=None, start_token=""):
    if prefixes is None:
        prefixes = get_prefixes(sample, start_token)

    model, states, delta, final_states, x = setup_model(dfa_size, alphabet, prefixes)
    add_transition_constraints(model, states, alphabet, delta)
    add_prefix_constraints(model, states, prefixes, x)
    add_initial_state_constraint(model, x, "")
    add_transition_follow_constraints(model, states, prefixes, alphabet, x, delta)
    alpha = add_alpha_constraints(model, states, sample, x, final_states)

    return model, states, delta, final_states, alpha


def learn_minimal_dfa(sample, alphabet, lower_bound, upper_bound, min_dfa_size=1, start_token=""):
    prefixes, max_dfa_size = initialize_prefixes_and_bounds(sample,
                                                            alphabet,
                                                            lower_bound,
                                                            upper_bound,
                                                            min_dfa_size,
                                                            start_token)
    for dfa_size in range(min_dfa_size, max_dfa_size):
        model, states, delta, final_states, alpha = set_dfa(sample, alphabet, dfa_size, prefixes, start_token)
        add_sample_constraints(model, states, sample, alpha, lower_bound, upper_bound)
        set_bounded_objective(model, states, sample, alpha, lower_bound, upper_bound)
        model.optimize()
        solution = extract_dfa_solution(model, states, alphabet, delta, final_states)
        if solution:
            return solution
    return None


def add_accepted_constraints(model, states, sample, alpha):
    """Add constraints on accepted"""
    accepted = model.addVars(sample, vtype=GRB.BINARY, name="accepted")
    for w in sample:
        model.addConstr(accepted[w] <= sum(alpha[w, q] for q in states), f"accept_{w}_bound_upper")
        for q in states:
            model.addConstr(accepted[w] >= alpha[w, q], f"accept_{w}_bound_lower_{q}")
    return accepted


def add_gamma_constraints(model, sample, accepted):
    """Add constraints for prefix acceptance."""
    gamma = model.addVars(((s, s_) for s in sample for s_ in sample), vtype=GRB.BINARY, name="gammas")
    for w in sample:
        for w_ in sample:
            if w == w_:
                continue
            model.addConstr(gamma[w, w_] >= accepted[w] - accepted[w_], f"accept_and_rejected_{w}_{w_}_1")
            model.addConstr(gamma[w, w_] <= accepted[w], f"accept_and_rejected_{w}_{w_}_2")
            model.addConstr(gamma[w, w_] <= 1 - accepted[w_], f"accept_and_rejected_{w}_{w_}_3")
    return gamma


def add_distance_based_objective(model, sample, accepted, gamma, distance_matrix):
    """
    Add an objective to a Gurobi model based on distances between sequences.

    Parameters:
    - model: Gurobi model.
    - sample: List of sequences in the sample.
    - accepted: Dictionary of accepted variables for each sequence index.
    - gamma: Dictionary of gamma variables for sequence pairs.
    - distance_matrix: Precomputed distance matrix (numpy array).
    """
    n = len(sample)
    max_dist = find_max_dist(distance_matrix)
    model.setObjective(
        sum(
            accepted[sample[i]] * distance_matrix[i, j] + max_dist * gamma[sample[i], sample[j]]
            for i in range(n) for j in range(n)
        ),
        GRB.MINIMIZE
    )


def learn_distance_based_dfa(sample, alphabet, dfa_size, start_token=""):
    prefixes = get_prefixes(sample, start_token)
    model, states, delta, final_states, alpha = set_dfa(sample, alphabet, dfa_size, prefixes, start_token)
    accepted = add_accepted_constraints(model, states, sample, alpha)
    gamma = add_gamma_constraints(model, sample, accepted)

    distance_matrix = calculate_distance_matrix(sample, levenshtein_distance)
    add_distance_based_objective(model, sample, accepted, gamma, distance_matrix)

    model.optimize()
    solution = extract_dfa_solution(model, states, alphabet, delta, final_states)
    if solution:
        return solution


# Example usage
sample = [("a", "b", "b"), ("a", "a"), ("a", "b"), ("a", "b", "b", "c", "c"), ("a", "a", "b"), ("b", "a", "b"), ("a", "a", "a"),
          ("c", "c", "c", "c", "c", "a", "b"), ("c", "c", "c", "c", "c", "c", "c", "c", "a")]
alphabet = {"a", "b", "c"}
lower_bound = 1
upper_bound = 1

#dfa = learn_minimal_dfa(sample, alphabet, lower_bound, upper_bound, min_dfa_size=1)
dfa = learn_distance_based_dfa(sample, alphabet, dfa_size=10)


if dfa:
    print("Minimal DFA found:")
    print(f"States: {dfa['states']}")
    print(f"Alphabet: {dfa['alphabet']}")
    print(f"Transitions: {dfa['transitions']}")
    print(f"Initial State: {dfa['initial_state']}")
    print(f"Final States: {dfa['final_states']}")
else:
    print("No feasible DFA found within the given bounds.")


def evaluate_sample_with_dfa(sample, dfa):
    accepted = []
    rejected = []

    transitions = dfa['transitions']
    initial_state = dfa['initial_state']
    final_states = dfa['final_states']

    for word in sample:
        current_state = initial_state
        accepted_word = True

        for symbol in word:
            if (current_state, symbol) in transitions:
                current_state = transitions[(current_state, symbol)]
            else:
                accepted_word = (current_state in final_states)
                break

        if accepted_word and current_state in final_states:
            accepted.append(word)
        else:
            rejected.append(word)

    return accepted, rejected

accepted, rejected = evaluate_sample_with_dfa(sample, dfa)

print("Accepted words:", accepted)
print("Rejected words:", rejected)
