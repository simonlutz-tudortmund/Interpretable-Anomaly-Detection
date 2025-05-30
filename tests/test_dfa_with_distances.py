from src.milp.milp_data_file import (
    learn_dfa_with_distances_linear,
    learn_dfa_with_distances_quadratic,
)
import pytest
import random

from src.utils.distance_functions import (
    calculate_distance_matrix,
    distance_function_names,
)


class TestDFALearningWithDistances:
    @pytest.fixture
    def sample_data(self):
        """
        Fixture to provide sample data for tests.

        Generates a random set of traces where approximately 20% are guaranteed
        to be easily separatable from the rest based on a very distinct pattern.

        Returns:
            tuple: (sample, alphabet) where sample is a list of traces and
                alphabet is the set of symbols used.
        """
        # Define our alphabet
        alphabet = {"a", "b", "c"}

        # Generate base random traces
        base_traces = []
        num_traces = 10  # Total number of traces

        # Generate 80% random traces that never contain "cc"
        for _ in range(int(num_traces * 0.8)):
            # Random length between 1 and 5
            length = random.randint(1, 5)

            # Generate trace ensuring no "cc" sequence
            trace = []
            last_char = None
            for _ in range(length):
                # If last character was 'c', don't allow another 'c'
                if last_char == "c":
                    next_char = random.choice(["a", "b"])
                else:
                    next_char = random.choice(list(alphabet))
                trace.append(next_char)
                last_char = next_char

            base_traces.append(tuple(trace))

        # Generate 20% separatable traces - all containing "cc" sequence
        separatable_traces = []
        for _ in range(int(num_traces * 0.2)):
            # Random length between 3 and 6 (to ensure room for "cc" plus other chars)
            length = random.randint(3, 6)

            # Create a trace with "cc" somewhere in the middle
            position = random.randint(0, length - 2)  # Position to insert "cc"

            trace = []
            for i in range(length):
                if i == position:
                    trace.append("c")
                elif i == position + 1:
                    trace.append("c")
                else:
                    trace.append(random.choice(list(alphabet)))

            separatable_traces.append(tuple(trace))

        # Combine and shuffle all traces
        random.shuffle(base_traces)
        random.shuffle(separatable_traces)

        return base_traces, separatable_traces, alphabet

    @pytest.mark.parametrize(
        "distance_function",
        [
            "levenshtein",
            "jaccard",
            # "hamming",
            # "ordinal_hamming",
            # "dynamic_time_warping",
        ],
    )
    def test_distances_quadratic(self, sample_data, distance_function):
        distance_function = distance_function
        base_traces, separatable_traces, alphabet = sample_data
        sample = base_traces + separatable_traces
        random.shuffle(sample)

        dfa, _ = learn_dfa_with_distances_quadratic(
            sample=sample,
            alphabet=alphabet,
            distance_function=distance_function,
            dfa_size=3,
            lambda_l=None,
            lambda_s=None,
            lambda_p=None,
            verbose=2,
        )
        # print(dfa)
        # assert all(base_trace not in dfa for base_trace in base_traces), (
        #     "All base traces should be accepted by the DFA."
        # )
        # assert all(
        #     separatable_trace in dfa for separatable_trace in separatable_traces
        # ), "No separatable traces should be accepted by the DFA."

        # Get accepted and rejected samples
        accepted_sample = [trace for trace in sample if trace in dfa]
        rejected_sample = [trace for trace in sample if trace not in dfa]

        # Calculate distance matrix for the sample
        distance_matrix = calculate_distance_matrix(
            sample=sample,
            distance_func=distance_function_names[distance_function],
        )

        # Calculate objective value for the current partition
        current_objective = calculate_partition_objective(
            sample=sample,
            accepted=accepted_sample,
            rejected=rejected_sample,
            distance_matrix=distance_matrix,
        )

        # Iterate over all possible partitions with the same accepted count
        num_accepted = len(accepted_sample)
        n = len(sample)
        max_objective = current_objective
        best_partition = None

        import itertools

        # Generate all possible combinations of accepted traces with the same length
        for accepted_indices in itertools.combinations(range(n), num_accepted):
            candidate_accepted = [sample[i] for i in accepted_indices]
            candidate_rejected = [
                sample[i] for i in range(n) if i not in accepted_indices
            ]

            objective = calculate_partition_objective(
                sample=sample,
                accepted=candidate_accepted,
                rejected=candidate_rejected,
                distance_matrix=distance_matrix,
            )

            if objective > max_objective:
                max_objective = objective
                best_partition = (candidate_accepted, candidate_rejected)

        # Assert that our DFA found the optimal partition
        assert current_objective == max_objective, (
            f"DFA partition objective ({current_objective}) is not equal to the optimal objective ({max_objective})"
        )

        # Print information if the test fails
        if current_objective < max_objective and best_partition:
            print("DFA's partition is not optimal.")
            print(f"DFA accepted: {accepted_sample}")
            print(f"DFA rejected: {rejected_sample}")
            print(f"Best accepted: {best_partition[0]}")
            print(f"Best rejected: {best_partition[1]}")

    @pytest.mark.parametrize(
        "distance_function",
        [
            "levenshtein",
            "jaccard",
            # "hamming",
            # "ordinal_hamming",
            # "dynamic_time_warping",
        ],
    )
    def test_distances_linear(self, sample_data, distance_function):
        distance_function = distance_function
        base_traces, separatable_traces, alphabet = sample_data
        sample = base_traces + separatable_traces
        random.shuffle(sample)

        dfa, _ = learn_dfa_with_distances_linear(
            sample=sample,
            alphabet=alphabet,
            distance_function=distance_function,
            dfa_size=4,
            lambda_l=None,
            lambda_s=None,
            lambda_p=None,
            verbose=2,
        )
        # print(dfa)

        # assert all(base_trace not in dfa for base_trace in base_traces), (
        #     "All base traces should be accepted by the DFA."
        # )
        # assert all(
        #     separatable_trace in dfa for separatable_trace in separatable_traces
        # ), "No separatable traces should be accepted by the DFA."

        # Get accepted and rejected samples
        accepted_sample = [trace for trace in sample if trace in dfa]
        rejected_sample = [trace for trace in sample if trace not in dfa]

        # Calculate distance matrix for the sample
        distance_matrix = calculate_distance_matrix(
            sample=sample,
            distance_func=distance_function_names[distance_function],
        )

        # Calculate objective value for the current partition
        current_objective = calculate_partition_objective(
            sample=sample,
            accepted=accepted_sample,
            rejected=rejected_sample,
            distance_matrix=distance_matrix,
        )

        # Iterate over all possible partitions with the same accepted count
        num_accepted = len(accepted_sample)
        n = len(sample)
        max_objective = current_objective
        best_partition = None

        import itertools

        # Generate all possible combinations of accepted traces with the same length
        for accepted_indices in itertools.combinations(range(n), num_accepted):
            candidate_accepted = [sample[i] for i in accepted_indices]
            candidate_rejected = [
                sample[i] for i in range(n) if i not in accepted_indices
            ]

            objective = calculate_partition_objective(
                sample=sample,
                accepted=candidate_accepted,
                rejected=candidate_rejected,
                distance_matrix=distance_matrix,
            )

            if objective > max_objective:
                max_objective = objective
                best_partition = (candidate_accepted, candidate_rejected)

        # Assert that our DFA found the optimal partition
        assert current_objective == max_objective, (
            f"DFA partition objective ({current_objective}) is not equal to the optimal objective ({max_objective})"
        )

        # Print information if the test fails
        if current_objective < max_objective and best_partition:
            print("DFA's partition is not optimal.")
            print(f"DFA accepted: {accepted_sample}")
            print(f"DFA rejected: {rejected_sample}")
            print(f"Best accepted: {best_partition[0]}")
            print(f"Best rejected: {best_partition[1]}")


def calculate_partition_objective(sample, accepted, rejected, distance_matrix):
    """Calculate the objective value for a given partition of accepted and rejected traces."""
    # Map traces to their indices in the sample
    trace_to_index = {trace: i for i, trace in enumerate(sample)}

    # Sum of distances between rejected and accepted traces
    between_sum = 0
    for rejected_trace in rejected:
        for accepted_trace in accepted:
            i = trace_to_index[rejected_trace]
            j = trace_to_index[accepted_trace]
            between_sum += distance_matrix[i][j]

    # Sum of distances between rejected traces
    within_sum = 0
    for i, rejected_trace1 in enumerate(rejected):
        for rejected_trace2 in rejected[i + 1 :]:
            i = trace_to_index[rejected_trace1]
            j = trace_to_index[rejected_trace2]
            within_sum += distance_matrix[i][j]

    # Return objective: maximize distance between classes, minimize within rejected class
    return between_sum - within_sum
