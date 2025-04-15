from src.milp.milp_data_file import learn_dfa_with_labels
import pytest
import random


class TestDFALearningWithLabels:
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

    def test_labels(self, sample_data):
        negative_sample, positive_sample, alphabet = sample_data

        dfa = learn_dfa_with_labels(
            positive_sample=positive_sample,
            negative_sample=negative_sample,
            alphabet=alphabet,
            min_dfa_size=1,
            lambda_l=None,
            lambda_s=None,
            lambda_p=None,
            verbose=2,
        )

        # Verify bounds are respected
        assert all(positive_trace in dfa for positive_trace in positive_sample)
        assert all(negative_trace not in dfa for negative_trace in negative_sample)

    @pytest.mark.parametrize(
        "penalty_type",
        [
            [],
            ["sink"],
            ["self_loop"],
            ["parallel"],
            ["sink", "self_loop"],
            ["sink", "parallel"],
            ["self_loop", "parallel"],
            ["sink", "self_loop", "parallel"],
        ],
    )
    def test_combined_penalties(self, sample_data, penalty_type):
        """Comprehensive test of all penalties with different bound configurations."""
        negative_sample, positive_sample, alphabet = sample_data

        # Set up the specific penalty
        lambda_s = 0.7 if penalty_type == "sink" else None
        lambda_l = 0.7 if penalty_type == "self_loop" else None
        lambda_p = 0.7 if penalty_type == "parallel" else None

        dfa = learn_dfa_with_labels(
            positive_sample=positive_sample,
            negative_sample=negative_sample,
            alphabet=alphabet,
            min_dfa_size=1,
            lambda_l=lambda_l,
            lambda_s=lambda_s,
            lambda_p=lambda_p,
            verbose=2,
        )

        # Verify bounds are respected
        assert all(positive_trace in dfa for positive_trace in positive_sample)
        assert all(negative_trace not in dfa for negative_trace in negative_sample)
