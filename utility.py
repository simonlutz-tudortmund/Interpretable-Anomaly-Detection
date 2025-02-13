from typing import List, Tuple


def get_prefixes(sample, start_token=""):
    prefixes = {(start_token,)} | {tuple(pref) for word in sample for pref in
                                   (tuple(word[:i + 1]) for i in range(len(word)))}
    return prefixes


def read_sample_from_file(filepath: str, delimiter: str = ";") -> Tuple[List[Tuple[str, ...]], set]:
    """
    Read a sample from a file where each line is one word and individual letters are separated by a delimiter.
    """
    sample = []
    alphabet = set()
    with open(filepath, 'r') as file:
        for line in file:
            word = tuple(line.strip().split(delimiter))
            sample.append(word)
            alphabet.update(word)
    return sample, alphabet
