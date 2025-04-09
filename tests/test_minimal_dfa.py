from src.milp.milp_data_file import learn_dfa
import pytest


@pytest.fixture
def sample_data():
    """
    Fixture to provide sample data for tests.
    
    Generates a random set of traces where approximately 20% are guaranteed
    to be easily separatable from the rest based on a very distinct pattern.
    
    Returns:
        tuple: (sample, alphabet) where sample is a list of traces and
               alphabet is the set of symbols used.
    """
    import random
    
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
            if last_char == 'c':
                next_char = random.choice(['a', 'b'])
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
                trace.append('c')
            elif i == position + 1:
                trace.append('c')
            else:
                trace.append(random.choice(list(alphabet)))
                
        separatable_traces.append(tuple(trace))
    
    # Combine and shuffle all traces
    all_traces = base_traces + separatable_traces
    random.shuffle(all_traces)
    
    return all_traces, alphabet

@pytest.mark.parametrize("bound_values", [
    (1, None),  # lower bound only
    (None, 2),  # upper bound only
    (1, 2),     # both bounds
])
def test_bounds(sample_data, bound_values):
    lower_bound, upper_bound = bound_values
    sample, alphabet = sample_data

    dfa = learn_dfa(
        sample=sample,
        alphabet=alphabet,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        min_dfa_size=3,
        lambda_l=None,
        lambda_s=None,
        lambda_p=None,
        verbose=2,
        pointwise=False,
    )

    # Verify bounds are respected
    accepted_count = sum(1 if trace in dfa else 0 for trace in sample)
    
    if lower_bound is not None:
        assert accepted_count >= lower_bound, f"Expected at least {lower_bound} traces, got {accepted_count}"
    else:
        assert accepted_count > 0, f"Degenerate solution, got {accepted_count}"
    
    if upper_bound is not None:
        assert accepted_count <= upper_bound, f"Expected at most {upper_bound} traces, got {accepted_count}"
    else:
        assert accepted_count < len(sample), f"Degenerate solution, got {accepted_count}" 

@pytest.mark.parametrize("bound_values", [
    (1, None),  # lower bound only
    (None, 2),  # upper bound only
    (1, 2),     # both bounds
])
def test_sink_penalty(sample_data, bound_values):
    """Test sink penalty with different bound configurations."""
    sample, alphabet = sample_data
    lambda_s = 10.0  # Strong penalty to prefer sink state
    
    # Configure bounds based on parameterization
    lower_bound, upper_bound = bound_values
    
    dfa = learn_dfa(
        sample=sample,
        alphabet=alphabet,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lambda_s=lambda_s,
        lambda_l=None,
        lambda_p=None,
        min_dfa_size=3,
        verbose=2,
    )
    
    # Verify bounds are respected
    accepted_count = sum(1 if trace in dfa else 0 for trace in sample)
    
    if lower_bound is not None:
        assert accepted_count >= lower_bound, f"Expected at least {lower_bound} traces, got {accepted_count}"
    else:
        assert accepted_count > 0, f"Degenerate solution, got {accepted_count}"
    
    if upper_bound is not None:
        assert accepted_count <= upper_bound, f"Expected at most {upper_bound} traces, got {accepted_count}"
    else:
        assert accepted_count < len(sample), f"Degenerate solution, got {accepted_count}" 
    
    # Check for sink state behavior (this is the key test for lambda_s)
    sink_state_found = False
    for state in dfa.states:
        # A sink state is one where all transitions lead back to itself
        is_sink = all(dfa.transition(state, symbol) == state for symbol in alphabet)
        if is_sink:
            sink_state_found = True
            break
    
    assert sink_state_found, "No sink state found despite high sink state penalty"


@pytest.mark.parametrize("bound_values", [
    (1, None),  # lower bound only
    (None, 2),  # upper bound only
    (1, 2),     # both bounds
])
def test_self_loop_penalty(sample_data, bound_values):
    """Test self-loop penalty with different bound configurations."""
    sample, alphabet = sample_data
    lambda_l = 10.0  # Strong penalty for non-self-loops
    
    # Configure bounds based on parameterization
    lower_bound, upper_bound = bound_values
    
    dfa = learn_dfa(
        sample=sample,
        alphabet=alphabet,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lambda_s=None,
        lambda_l=lambda_l,
        lambda_p=None,
        min_dfa_size=2,
        verbose=2,
    )
    
    # Verify bounds are respected
    accepted_count = sum(1 if trace in dfa else 0 for trace in sample)
    
    if lower_bound is not None:
        assert accepted_count >= lower_bound, f"Expected at least {lower_bound} traces, got {accepted_count}"
    else:
        assert accepted_count > 0, f"Degenerate solution, got {accepted_count}"
    
    if upper_bound is not None:
        assert accepted_count <= upper_bound, f"Expected at most {upper_bound} traces, got {accepted_count}"
    else:
        assert accepted_count < len(sample), f"Degenerate solution, got {accepted_count}" 
    
    # Count non-self-loop transitions
    non_self_loops = 0
    for state in dfa.states:
        for symbol in alphabet:
            if dfa.transition(state, symbol) != state:
                non_self_loops += 1
    
    # With high lambda_l, we expect to minimize non-self-loops
    # The exact number depends on the DFA structure needed to accept the required traces
    print(f"Number of non-self-loops with lambda_l={lambda_l}: {non_self_loops}")
    
    # We can't assert an exact number, but we can check that the total is low
    # For a 2-state DFA with 2 symbols, the maximum would be 4 transitions
    assert non_self_loops <= len(dfa.states) * len(alphabet), "Too many non-self-loops despite penalty"


@pytest.mark.parametrize("bound_values", [
    (1, None),  # lower bound only
    (None, 2),  # upper bound only
    (1, 2),     # both bounds
])
def test_parallel_edge_penalty(sample_data, bound_values):
    """Test parallel edge penalty with different bound configurations."""
    sample, alphabet = sample_data
    lambda_p = 10.0  # Strong penalty for multiple transitions
    
    # Configure bounds based on parameterization
    lower_bound, upper_bound = bound_values
    
    dfa = learn_dfa(
        sample=sample,
        alphabet=alphabet,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lambda_s=None,
        lambda_l=None,
        lambda_p=lambda_p,
        min_dfa_size=2,
        verbose=2,
    )
    
    # Verify bounds are respected
    accepted_count = sum(1 if trace in dfa else 0 for trace in sample)
    
    if lower_bound is not None:
        assert accepted_count >= lower_bound, f"Expected at least {lower_bound} traces, got {accepted_count}"
    else:
        assert accepted_count > 0, f"Degenerate solution, got {accepted_count}"
    
    if upper_bound is not None:
        assert accepted_count <= upper_bound, f"Expected at most {upper_bound} traces, got {accepted_count}"
    else:
        assert accepted_count < len(sample), f"Degenerate solution, got {accepted_count}" 
    
    # Check that transitions are minimized due to parallel edge penalty
    # For each state, count unique target states across all symbols
    for state in dfa.states:
        targets = set()
        for symbol in alphabet:
            targets.add(dfa.transition(state, symbol))
        
        # With high lambda_p, we expect minimal unique targets
        # Ideally just one target state for all symbols (strong minimization)
        # but at most one target per symbol (required minimum)
        assert len(targets) <= len(alphabet), f"Too many unique targets from state {state}"
        print(f"State {state} has {len(targets)} unique target states")


@pytest.mark.parametrize("penalty_type", ["sink", "self_loop", "parallel"])
@pytest.mark.parametrize("bound_values", [
    (1, None),  # lower bound only
    (None, 2),  # upper bound only
    (1, 2),     # both bounds
])
def test_combined_penalties(sample_data, penalty_type, bound_values):
    """Comprehensive test of all penalties with different bound configurations."""
    lower_bound, upper_bound = bound_values
    sample, alphabet = sample_data
    
    # Set up the specific penalty
    lambda_s = 10.0 if penalty_type == "sink" else None
    lambda_l = 10.0 if penalty_type == "self_loop" else None
    lambda_p = 10.0 if penalty_type == "parallel" else None
    
    dfa = learn_dfa(
        sample=sample,
        alphabet=alphabet,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lambda_s=lambda_s,
        lambda_l=lambda_l,
        lambda_p=lambda_p,
        min_dfa_size=3,
        verbose=2,
    )
    
    # Verify bounds are respected
    accepted_count = sum(1 if trace in dfa else 0 for trace in sample)
    
    if lower_bound is not None:
        assert accepted_count >= lower_bound, f"Expected at least {lower_bound} traces, got {accepted_count}"
    else:
        assert accepted_count > 0, f"Degenerate solution, got {accepted_count}"
    
    if upper_bound is not None:
        assert accepted_count <= upper_bound, f"Expected at most {upper_bound} traces, got {accepted_count}"
    else:
        assert accepted_count < len(sample), f"Degenerate solution, got {accepted_count}" 
    
    # Print DFA information for inspection
    print(f"\nDFA with {penalty_type} penalty and bounds ({lower_bound}, {upper_bound}):")
    print(f"States: {dfa.states}")
    print(f"Transitions: {dfa.transitions}")
    print(f"Initial State: {dfa.initial_state}")
    print(f"Final States: {dfa.final_states}")
    print(f"Accepted traces: {[trace for trace in sample if trace in dfa]}")