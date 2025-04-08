from typing import Any
from gurobipy import Model, GRB, Var, quicksum, tupledict


class ClassificationNode:
    """Represents a node in the prefix tree with reachability variables."""

    def __init__(self, reach_vars):
        self.reach_vars = reach_vars  # List of Gurobi variables for each state
        self.children = {}  # Mapping from letters to child nodes


class GurobiClassification:
    """Manages the prefix tree structure and constraints for the MILP model."""

    def __init__(
        self,
        model: Model,
        num_states: int,
        alphabet: frozenset[str],
        trans_vars: tupledict[int, Var],
        term_vars: tupledict[int, Var],
    ):
        self.model = model
        self.N = num_states
        self.alphabet = alphabet
        self.trans_vars = trans_vars  # trans_vars[p][a_idx][q]
        self.term_vars = term_vars  # term_vars[q]
        self.root = self._create_root_node()

    def _create_root_node(self):
        """Create the root node (empty word) with initial state constraints."""
        reach_vars = [self.model.addVar(vtype=GRB.BINARY) for q in range(self.N)]
        # Exactly one state is active
        self.model.addConstr(quicksum(reach_vars) == 1, "root_sum")
        # Initial state is 0
        self.model.addConstr(reach_vars[0] == 1, "initial_state")
        return ClassificationNode(reach_vars)

    def _create_new_node(self):
        """Create a new node with reachability variables and sum constraint."""
        reach_vars = [self.model.addVar(vtype=GRB.BINARY) for q in range(self.N)]
        self.model.addConstr(quicksum(reach_vars) == 1)
        return ClassificationNode(reach_vars)

    def _add_transition_constraints(self, parent_reach, child_reach, a_idx):
        """Add constraints for transitions from parent to child node on letter a."""
        for p in range(self.N):
            for q in range(self.N):
                # Create a constraint: parent_reach[p] => (child_reach[q] == trans_vars[p][a_idx][q])
                # This can be expressed as two implications:
                # 1. parent_reach[p] and trans_vars[p,a_idx,q] => child_reach[q]
                # 2. parent_reach[p] and not trans_vars[p,a_idx,q] => not child_reach[q]

                # First part: if parent is in p and trans is 1, child must be in q
                self.model.addGenConstrIndicator(
                    parent_reach[p],
                    True,
                    child_reach[q] >= self.trans_vars[p, a_idx, q],
                    name=f"trans_ind_pos_{p}_{a_idx}_{q}",
                )

                # Second part: if parent is in p and trans is 0, child must not be in q
                self.model.addGenConstrIndicator(
                    parent_reach[p],
                    True,
                    child_reach[q] <= self.trans_vars[p, a_idx, q],
                    name=f"trans_ind_neg_{p}_{a_idx}_{q}",
                )

    def generate_clauses_word(self, word: list[str]):
        """Build the prefix tree for the word and return the final node's reach variables."""
        current_node = self.root
        for letter in word:
            if letter not in current_node.children:
                a_idx = list(self.alphabet).index(letter)
                # Create new node with a unique name based on the prefix
                child_node = self._create_new_node()
                # Add transition constraints from current node to child node
                self._add_transition_constraints(
                    current_node.reach_vars, child_node.reach_vars, a_idx
                )
                current_node.children[letter] = child_node
            current_node = current_node.children[letter]
        return current_node.reach_vars

    def generate_clauses_trace(
        self, word: list[str], word_index: int, alpha: tupledict[Any, Var]
    ):
        """Add constraints for the word's acceptance based on its label."""
        reach_vars = self.generate_clauses_word(word)
        for q in range(self.N):
            self.model.addConstr(
                alpha[word_index, q] >= reach_vars[q] + self.term_vars[q] - 1
            )
            self.model.addConstr(alpha[word_index, q] <= reach_vars[q])
            self.model.addConstr(alpha[word_index, q] <= self.term_vars[q])
