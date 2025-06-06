
# DFA Learning Project

This project implements an algorithm for learning a minimal Deterministic Finite Automaton (DFA) using the Gurobi optimization solver. The approach is based on a set of constraints and objective functions designed to learn DFA models that accept or reject a given sample of words from an alphabet.

## Features

- **Minimal DFA Learning**: The core functionality is the learning of a minimal DFA for a given sample and alphabet.
- **Gurobi Optimization**: Uses the Gurobi optimization solver to find the DFA that satisfies transition and acceptance constraints.
- **Customizable Bounds**: The user can set lower and upper bounds on the number of accepted words in the DFA.
- **Prefix-based DFA construction**: The DFA is constructed using prefixes of the sample words to define states and transitions.
  
## Installation

To install the required dependencies for this project, run:

```
pip install poetry
poetry install
```

Make sure you have a Gurobi license, as the `gurobipy` library is the official Python interface to the Gurobi Optimizer.

## Usage

Here is an example of how to use the algorithm to learn a minimal DFA for a given set of sample words and an alphabet:

```python
sample = [("a", "b", "b"), ("a", "a"), ("a", "b"), ("b", "a"), ("a", "a", "b"), ("b", "a", "b"), ("a", "a", "a"),
          ("c", "c", "c", "c", "c"), ("c", "c", "c", "c", "c", "c")]
alphabet = {"a", "b", "c"}
lower_bound = 2
upper_bound = None

dfa, _ = learn_dfa_with_bounds(sample, alphabet, lower_bound, upper_bound, min_dfa_size=2)

if dfa:
    print("Minimal DFA found:")
    print(f"States: {dfa.states}")
    print(f"Alphabet: {dfa.alphabet}")
    print(f"Transitions: {dfa.transitions}")
    print(f"Initial State: {dfa.initial_state}")
    print(f"Final States: {dfa.final_states}")
else:
    print("No feasible DFA found within the given bounds.")
```

### Main Functions

- `learn_dfa_with_bounds(sample, alphabet, lower_bound, upper_bound, min_dfa_size=1)`: Learns the minimal DFA from the given sample and alphabet within the provided bounds.
- `learn_dfa_with_labels(positive_sample, negative_sample, alphabet, min_dfa_size=1)`: Learns the minimal DFA that accepts all words from positive_sample and rejects all negative_sample.
- `learn_dfa_with_distances_linear(sample, alphabet, distance_function, dfa_size=1)`: Learns a DFA from that maximizes distance between accepted and rejected words, while minimizing distance between rejected words using a combination of linear constraints.
- `learn_dfa_with_distances_quadratic(sample, alphabet, distance_function, dfa_size=1)`: Learns a DFA from that maximizes distance between accepted and rejected words, while minimizing distance between rejected words using multiplication of matrix variables.

### Constraints

- The algorithm ensures that the DFA satisfies various constraints such as:
  - A single transition for each state-symbol pair.
  - A unique state for each word prefix.
  - Acceptance constraints for the sample words.

### Example Output

```bash
Minimal DFA found:
States: [0, 1, 2, 3]
Alphabet: {'a', 'b', 'c'}
Transitions: {(0, 'a'): 1, (1, 'b'): 2, (2, 'c'): 3}
Initial State: 0
Final States: {3}
```
