from typing import List, Optional


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
