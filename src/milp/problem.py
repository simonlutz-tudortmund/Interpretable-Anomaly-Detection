import logging
import sys
from typing import Optional
from gurobipy import Model, GRB, quicksum

from src.milp.classification import GurobiClassification
from src.utils.dfa import DFABuilder


class Problem:
    def __init__(
        self,
        N: int,
        alphabet: frozenset[str],
    ):
        self.N: int = N
        self.alphabet: frozenset[str] = alphabet
        self.restart()

    def restart(self):
        self.model = Model(f"Hypothesis_{self.N}")
        self.generate_clauses_automaton()

    def generate_clauses_automaton(self):
        self.states = range(self.N)

        ########################################################################################################################
        #                                           Unique transitions:
        ########################################################################################################################
        self.transition = self.model.addVars(
            self.N, len(self.alphabet), self.N, vtype=GRB.BINARY, name="delta"
        )
        for p in range(self.N):
            for a, letter in enumerate(self.alphabet):
                trans_q = [self.transition[p, a, q] for q in range(self.N)]
                self.model.addConstr(quicksum(trans_q) == 1, f"trans_sum_{p}_{a}")

        ########################################################################################################################
        #                                           Accepting state self loops:
        ########################################################################################################################
        self.terminal = [
            self.model.addVar(vtype=GRB.BINARY, name=f"term_{q}") for q in range(self.N)
        ]

        for q in self.states:
            for a, letter in enumerate(self.alphabet):
                self.model.addConstr((self.terminal[q] <= self.transition[q, a, q]))

        ########################################################################################################################
        #                                           Prefix tree for words:
        ########################################################################################################################
        self.classifier = GurobiClassification(
            model=self.model,
            num_states=self.N,
            alphabet=self.alphabet,
            trans_vars=self.transition,
            term_vars=self.terminal,
        )

    def get_automaton(self):
        """
        Extract results as a DFA.
        .. warning:: ``solve`` should have been called successfully before.
        """
        builder = DFABuilder(alphabet=self.alphabet, initial_state=0)
        for p in range(self.N):
            for a, letter in enumerate(self.alphabet):
                for q in range(self.N):
                    if self.transition[p, a, q].X > 0.5:
                        if (
                            p in builder.transitions
                            and letter in builder.transitions[p]
                        ):
                            logging.error(
                                "WARNING: automaton not deterministic (too many transitions)",
                                file=sys.stderr,
                            )
                        builder.transition(p, letter, q)
                if (
                    p not in builder.transitions
                    and letter not in builder.transitions[p]
                ):
                    logging.error(
                        "WARNING: automaton not deterministic (missing transition)",
                        file=sys.stderr,
                    )
            if self.terminal[p].X > 0.5:
                builder.tag(p)
        return builder.build()

    def add_sample_constraints(
        self,
        sample: list[list[str]],
        lower_bound: Optional[int] = None,
        upper_bound: Optional[int] = None,
    ):
        """Add constraints for prefix acceptance."""
        self.alpha = self.model.addVars(
            len(sample), self.N, vtype=GRB.BINARY, name="alpha"
        )
        for idx, word in enumerate(sample):
            self.classifier.generate_clauses_trace(
                word=word, word_index=idx, alpha=self.alpha
            )

        ########################################################################################################################
        #                                           Bound Constraints:
        ########################################################################################################################
        if lower_bound:
            self.model.addConstr(
                self.alpha.sum() >= lower_bound,
                "lower_bound",
            )
        if upper_bound:
            self.model.addConstr(
                self.alpha.sum() <= upper_bound,
                "upper_bound",
            )

        ########################################################################################################################
        #                                           Optimization intial objective:
        ########################################################################################################################
        if lower_bound and upper_bound:
            self.model.setParam("MIPFocus", 1)
            self.model.setObjective(1, GRB.MINIMIZE)
        elif lower_bound:
            self.model.setObjective(
                self.alpha.sum(),
                GRB.MINIMIZE,
            )
        elif upper_bound:
            self.model.setObjective(-1 * self.alpha.sum(), GRB.MINIMIZE)

    def add_sink_state_penalty(self, lambda_s, sink_state_index=1):
        """Add penalty for transitions not leading to the sink state."""
        sink_state = self.states[sink_state_index]
        penalty = lambda_s * quicksum(
            1 - self.transition[q, a, sink_state]
            for q in self.states
            for a, _ in enumerate(self.alphabet)
        )
        self.model.setObjective(self.model.getObjective() + penalty, GRB.MINIMIZE)

    def add_self_loop_penalty(self, lambda_l):
        """Add penalty for transitions that are not self-loops."""
        penalty = lambda_l * sum(
            self.transition[q, a, q0]
            for q in self.states
            for a, _ in enumerate(self.alphabet)
            for q0 in self.states
            if q0 != q
        )
        self.model.setObjective(self.model.getObjective() + penalty, GRB.MINIMIZE)

    def add_parallel_edge_penalty(self, lambda_p):
        """Add penalty for multiple successor states."""

        eq = self.model.addVars(self.N, self.N, vtype=GRB.BINARY, name="eq")
        for q in self.states:
            for q0 in self.states:
                self.model.addConstr(
                    eq[q, q0]
                    <= sum(
                        self.transition[q, a, q0] for a, _ in enumerate(self.alphabet)
                    )
                )
                for a, _ in enumerate(self.alphabet):
                    self.model.addConstr(eq[q, q0] >= self.transition[q, a, q0])
        penalty = lambda_p * sum(eq[q, q0] for q in self.states for q0 in self.states)
        self.model.setObjective(self.model.getObjective() + penalty, GRB.MINIMIZE)
