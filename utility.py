import logging
from graphviz import Digraph
import os
from typing import List, Tuple, Optional, Set
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(true_labels, predicted_labels):
    """
    Calculate classification metrics for binary classification.

    Parameters:
    - true_labels: True labels of the test set.
    - predicted_labels: Predicted labels (either 0 or 1) of the test set.

    Returns:
    - A dictionary with accuracy, precision, recall, f1_score, and confusion matrix.
    """
    label_map = {'accept': 1, 'reject': 0}
    true_labels_numeric = [label_map[label] for label in true_labels]
    predicted_labels_numeric = [label_map[label] for label in predicted_labels]

    metrics = {
        "accuracy": accuracy_score(true_labels_numeric, predicted_labels_numeric),
        "precision": precision_score(true_labels_numeric, predicted_labels_numeric),
        "recall": recall_score(true_labels_numeric, predicted_labels_numeric),
        "f1_score": f1_score(true_labels_numeric, predicted_labels_numeric),
        "confusion_matrix": confusion_matrix(true_labels_numeric, predicted_labels_numeric),
    }

    return metrics


def log_metrics(metrics):
    """
    Log the classification metrics in a structured format.

    Parameters:
    - metrics: A dictionary containing the metrics such as accuracy, precision, recall, f1_score, and confusion_matrix.
    """
    logging.info("Classification Metrics:")
    logging.info(f"Accuracy: {metrics['accuracy']:.2f}")
    logging.info(f"Precision: {metrics['precision']:.2f}")
    logging.info(f"Recall: {metrics['recall']:.2f}")
    logging.info(f"F1-Score: {metrics['f1_score']:.2f}")
    logging.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")


def get_prefixes(sample, start_token=""):
    prefixes = {(start_token,)} | {tuple(pref) for word in sample for pref in
                                   (tuple(word[:i + 1]) for i in range(len(word)))}
    return prefixes


def read_sample_from_file(
    filepath: str,
    delimiter: str = ";",
    is_numeric: bool = False,
    numeric_precision: int = 2,
    is_labeled: bool = False,
    label_delimiter: str = ","
) -> Tuple[List[Tuple[str, ...]], Set[str], Optional[List[str]]]:
    """
    Reads a sample from a file where each line represents one word, with letters/numbers separated by a delimiter.
    Supports numerical values with rounding and labeled samples.

    Parameters:
    - filepath (str): Path to the input file.
    - delimiter (str): Delimiter used to separate symbols in words.
    - is_numeric (bool): If True, treats symbols as numbers and rounds them.
    - numeric_precision (int): Decimal places for rounding numeric values.
    - is_labeled (bool): If True, treats the last part of each line as a label.
    - label_delimiter (str): Delimiter that separates the word from the label.

    Returns:
    - sample (List[Tuple[str, ...]]): List of words (tuples of symbols).
    - alphabet (Set[str]): Unique set of symbols used in the words.
    - labels (Optional[List[str]]): List of labels if `is_labeled` is True, otherwise None.
    """
    sample = []
    alphabet = set()
    labels = [] if is_labeled else None

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if is_labeled:
                word_part, label = line.rsplit(label_delimiter, 1)
                labels.append(label.strip())
            else:
                word_part = line

            tokens = word_part.split(delimiter)
            if is_numeric:
                tokens = tuple(str(round(float(token), numeric_precision)) for token in tokens)
            else:
                tokens = tuple(tokens)

            sample.append(tokens)
            alphabet.update(tokens)

    if is_labeled:
        return sample, alphabet, labels
    return sample, alphabet


def load_sample(filepath: str, is_numeric: bool, numeric_precision: int = 2, is_labeled: bool = False):
    """Loads a sample from a file and returns the parsed sample, alphabet, and labels if applicable."""
    try:
        return read_sample_from_file(filepath, is_numeric=is_numeric, numeric_precision=numeric_precision, is_labeled=is_labeled)
    except FileNotFoundError:
        logging.critical(f"File {filepath} not found.")
        return None, None, None


def log_dfa_info(dfa: dict):
    """Logs information about the trained DFA."""
    logging.info("Minimal DFA found:")
    logging.info(f"States: {dfa['states']}")
    logging.info(f"Alphabet: {dfa['alphabet']}")
    logging.info(f"Transitions: {dfa['transitions']}")
    logging.info(f"Initial State: {dfa['initial_state']}")
    logging.info(f"Final States: {dfa['final_states']}")


def save_visualized_dfa(dfa, output_path="dfa"):
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
    # Remove extension if present
    if output_path.lower().endswith(".png"):
        output_path = os.path.splitext(output_path)[0]

    dot = Digraph(format="png")

    for state in dfa['states']:
        shape = "doublecircle" if state in dfa['final_states'] else "circle"
        dot.node(str(state), shape=shape)

    dot.node("start", shape="none", width="0")
    dot.edge("start", str(dfa['initial_state']))

    for (state, symbol), next_state in dfa['transitions'].items():
        dot.edge(str(state), str(next_state), label=str(symbol))

    dot.render(output_path, cleanup=True)
    logging.info(f"DFA visualization saved at {output_path}.png")


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
