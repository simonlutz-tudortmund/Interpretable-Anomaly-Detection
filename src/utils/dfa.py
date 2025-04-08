from typing import List, Optional, Set


class CausalDFA:
    def __init__(
        self,
        alphabet: frozenset[str],
        transitions: dict[int, dict[str, int]],
        final_states: frozenset[int],
        initial_state: int,
    ):
        self.initial_state = initial_state
        self.alphabet = alphabet
        self.transitions: dict[int, dict[str, int]] = transitions
        self.final_states: frozenset[int] = final_states
        all_states = set(self.transitions.keys()) | set(final_states) | {initial_state}
        for state_transitions in self.transitions.values():
            all_states.update(state_transitions.values())
        self.states: frozenset[int] = frozenset(all_states)

    def transition(self, current_state: int, input_symbol: str) -> int:
        if current_state in self.final_states:
            return current_state
        assert input_symbol in self.transitions[current_state]
        return self.transitions[current_state][input_symbol]

    def run(
        self,
        input_sequence: List[str],
        state: Optional[int] = None,
    ) -> bool:
        if state is None:
            current_state = self.initial_state
        else:
            current_state = state

        for input_item in input_sequence:
            current_state = self.transition(current_state, input_item)
        return current_state in self.final_states

    def __contains__(self, sequence: List[str]):
        """Check if a sequence is accepted by the DFA using the 'in' operator."""
        return self.run(input_sequence=sequence)

    def export_as_reward_automaton(self, file_path):
        return file_path

    def __eq__(self, other):
        """
        Check if this CausalDFA is equal to another CausalDFA.
        Two CausalDFAs are equal if they have the same alphabet set, transitions,
        final_states, and initial_state.
        """
        if not isinstance(other, CausalDFA):
            return False

        return (
            self.alphabet == other.alphabet
            and self.transitions == other.transitions
            and self.final_states == other.final_states
            and self.initial_state == other.initial_state
        )

    def __hash__(self):
        """
        Return a hash of this CausalDFA.
        """
        return hash(
            (
                self.alphabet,
                frozenset(
                    (k, frozenset(v.items())) for k, v in self.transitions.items()
                ),
                self.final_states,
                self.initial_state,
            )
        )

    def _find_paths(
        self, start: int, end: int, visited: Set[int], max_depth: int = 10
    ) -> List[List[tuple]]:
        """
        Find all possible simple paths from start state to end state, up to a maximum depth.

        Args:
            start: The starting state.
            end: The destination state.
            visited: Set of already visited states to avoid cycles.
            max_depth: Maximum depth of paths to explore to prevent infinite loops.

        Returns:
            List of paths, where each path is a list of (state, symbol, next_state) tuples.
        """
        if start == end:
            return [[]]  # Empty path if we're already at the end

        if start in visited or len(visited) >= max_depth:
            return []  # Prevent cycles or too deep paths

        visited_copy = visited.copy()
        visited_copy.add(start)

        all_paths = []

        # If the start state has transitions
        if start in self.transitions:
            for symbol, next_state in self.transitions[start].items():
                # Find all paths from next_state to end
                next_paths = self._find_paths(next_state, end, visited_copy, max_depth)

                # Prepend the current transition to each path
                for path in next_paths:
                    all_paths.append([(start, symbol, next_state)] + path)

        return all_paths


class DFABuilder:
    def __init__(self, alphabet: frozenset[str], initial_state: int = 0):
        self.alphabet = alphabet
        self.initial_state = initial_state
        self.final_states: set[int] = set()
        self.transitions: dict[int, dict[str, int]] = dict()

    def build(self) -> CausalDFA:
        dfa = CausalDFA(
            alphabet=self.alphabet,
            transitions=self.transitions,
            final_states=frozenset(self.final_states),
            initial_state=self.initial_state,
        )
        for state in dfa.states - dfa.final_states:
            for possible_transition in self.alphabet:
                if possible_transition not in self.transitions.get(state, {}):
                    self.transitions[state][possible_transition] = state
        return dfa

    def tag(self, state: int) -> "DFABuilder":
        self.final_states.add(state)
        return self

    def transition(
        self, from_state: int, input_symbol: str, to_state: int
    ) -> "DFABuilder":
        if from_state not in self.transitions:
            self.transitions[from_state] = dict()
        assert input_symbol not in self.transitions[from_state]
        self.transitions[from_state][input_symbol] = to_state
        return self
