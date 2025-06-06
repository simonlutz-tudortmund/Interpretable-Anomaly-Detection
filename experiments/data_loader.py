import re
from typing import Tuple, Set
import pandas as pd
from utils.paths import get_data_path


def load_alfred_data(goal: int) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    Load the dataset based on the provided dataset name and extract the alphabet.
    """

    def parse_features(features: str) -> Tuple[str, ...]:
        # Convert the string representation of a list to a tuple of strings
        return tuple(features.strip("[]").replace(" ", "").split(","))

    train_df = pd.read_csv(
        get_data_path().joinpath("Alfred", f"train_10{goal}.txt"),
        delimiter=";",  # Use semicolon as delimiter
        names=["Features", "Label"],  # Set column names
        converters={"Features": parse_features},  # Convert Features to tuple of strings
    )
    test_df = pd.read_csv(
        get_data_path().joinpath("Alfred", f"test_10{goal}.txt"),
        delimiter=";",  # Use semicolon as delimiter
        names=["Features", "Label"],  # Set column names
        converters={"Features": parse_features},  # Convert Features to tuple of strings
    )

    # Extract the alphabet as all unique strings in the "Features" column
    alphabet = set(
        feature
        for features_tuple in pd.concat([train_df["Features"], test_df["Features"]])
        for feature in features_tuple
    )

    return train_df, test_df, alphabet


def load_hfds_data(
    seed: int, test_percentage: float, max_sequence_length: int
) -> Tuple[pd.DataFrame, pd.DataFrame, frozenset[str]]:
    """
    Load the HFDS dataset based on the provided seed and split percentage.
    Ensure that both train and test splits contain every letter of the alphabet.
    """

    def convert_to_tuple(features_str):
        elements = re.findall(r"E\d+", features_str)
        return tuple(elements)

    # Load the dataset
    df = pd.read_csv(
        get_data_path().joinpath("HFDS", "HFDS_v1.csv"),  # Update path as needed
        index_col=0,
    )
    df["Features"] = df["Features"].apply(convert_to_tuple)

    # Filter by max sequence length
    df = df[df["Features"].apply(len) <= max_sequence_length]

    # Assuming labels are in a 'Label' column; adjust if different
    # If labels are mixed in 'Features', this needs different handling
    if "Label" in df.columns:
        df["Label"] = df["Label"].map({"Success": 0, "Fail": 1})
    else:
        raise ValueError("Label column not found. Check dataset structure.")

    # Extract alphabet from features
    alphabet = frozenset(value for trace in df["Features"] for value in trace)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Initial split
    split_index = int(len(df) * (1 - test_percentage))
    train_df = df[:split_index]
    test_df = df[split_index:]

    return train_df, test_df, alphabet


def load_bgl_data(
    seed: int, test_percentage: float, max_sequence_length: int
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    """
    Load the HFDS dataset based on the provided seed and split percentage.
    Ensure that both train and test splits contain every letter of the alphabet.
    """

    def convert_to_tuple(features_str):
        elements = re.findall(r"E\d+", features_str)
        return tuple(elements)

    # Load the dataset
    df = pd.read_csv(
        get_data_path().joinpath("BGL", "BGL.csv"),  # Update path as needed
        index_col=0,
    )
    df["Features"] = df["Features"].apply(convert_to_tuple)

    # Filter by max sequence length
    df = df[df["Features"].apply(len) <= max_sequence_length]

    # Assuming labels are in a 'Label' column; adjust if different
    # If labels are mixed in 'Features', this needs different handling
    df["Label"] = df["Label"].map({"Success": 0, "Fail": 1})

    # Extract alphabet from features
    alphabet = frozenset(value for trace in df["Features"] for value in trace)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Initial split
    split_index = int(len(df) * (1 - test_percentage))
    train_df = df[:split_index]
    test_df = df[split_index:]

    return train_df, test_df, alphabet


def load_openstack_data(
    seed: int, test_percentage: float, max_sequence_length: int
) -> Tuple[pd.DataFrame, pd.DataFrame, frozenset[str]]:
    """
    Load the HFDS dataset based on the provided seed and split percentage.
    Ensure that both train and test splits contain every letter of the alphabet.
    """

    def convert_to_tuple(features_str):
        elements = re.findall(r"E\d+", features_str)
        return tuple(elements)

    # Load the dataset
    df = pd.read_csv(
        get_data_path().joinpath("OpenStack", "Openstack.csv"),  # Update path as needed
        index_col=0,
    )
    df["Features"] = df["Features"].apply(convert_to_tuple)

    # Filter by max sequence length
    df = df[df["Features"].apply(len) <= max_sequence_length]

    # Assuming labels are in a 'Label' column; adjust if different
    # If labels are mixed in 'Features', this needs different handling
    if "Label" in df.columns:
        df["Label"] = df["Label"].map({"Success": 0, "Fail": 1})
    else:
        raise ValueError("Label column not found. Check dataset structure.")

    # Extract alphabet from features
    alphabet = frozenset(value for trace in df["Features"] for value in trace)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Initial split
    split_index = int(len(df) * (1 - test_percentage))
    train_df = df[:split_index]
    test_df = df[split_index:]

    return train_df, test_df, alphabet
