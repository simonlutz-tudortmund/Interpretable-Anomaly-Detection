from src.milp.milp_data_file import learn_dfa_with_distances
import pytest
import random


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
        num_traces = 5  # Total number of traces

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
        all_traces = base_traces + separatable_traces
        random.shuffle(all_traces)

        return all_traces, alphabet

    @pytest.mark.parametrize(
        "distance_function",
        [
            "levenshtein",
            "jaccard",
            "hamming",
            "ordinal_hamming",
            "dynamic_time_warping",
        ],
    )
    def test_distances(self, sample_data, distance_function):
        distance_function = distance_function
        sample, alphabet = sample_data

        dfa = learn_dfa_with_distances(
            sample=sample,
            alphabet=alphabet,
            distance_function=distance_function,
            dfa_size=3,
            lambda_l=None,
            lambda_s=None,
            lambda_p=None,
            verbose=2,
            pointwise=False,
        )

        # Verify bounds are respected
        accepted_count = sum(1 if trace in dfa else 0 for trace in sample)
        print(dfa)
