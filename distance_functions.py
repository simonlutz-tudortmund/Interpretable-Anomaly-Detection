import editdistance
from typing import List, Callable, Tuple
import numpy as np


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


def hamming_distance(seq1: Tuple[str, ...], seq2: Tuple[str, ...]) -> float:
    """
    Compute Hamming distance between two sequences.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    return sum(el1 != el2 for el1, el2 in zip(seq1, seq2))


def find_max_dist(distance_matrix: np.ndarray) -> float:
    """Find the maximum distance in the distance matrix."""
    return np.max(distance_matrix)


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
