import logging
import sys
from typing import Optional
from gurobipy import Model, GRB, quicksum
import numpy as np
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

    def add_automaton_constraints(self):
        self.states = range(self.N)

        ########################################################################################################################
        #                                           Unique transitions:
        ########################################################################################################################
        self.transition = self.model.addVars(
            self.N,
            len(self.alphabet),
            self.N,
            vtype=GRB.BINARY,
            name="delta",
        )
        # for var in self.transition.values():
        #     var.BranchPriority = 1
        for p in range(self.N):
            for a, letter in enumerate(self.alphabet):
                trans_q = [self.transition[p, a, q] for q in range(self.N)]
                self.model.addLConstr(quicksum(trans_q) == 1, f"trans_sum_{p}_{a}")

        ########################################################################################################################
        #                                           Accepting state self loops:
        ########################################################################################################################
        self.terminal = self.model.addVars(self.N, vtype=GRB.BINARY)

        # for var in self.terminal.values():
        #     var.BranchPriority = 2

        for q in self.states:
            for a, letter in enumerate(self.alphabet):
                self.model.addLConstr((self.terminal[q] <= self.transition[q, a, q]))

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

    def add_labeled_sample_constraints(
        self,
        positive_sample: list[tuple[str, ...]],
        negative_sample: list[tuple[str, ...]],
    ):
        for idx, word in enumerate(positive_sample):
            self.classifier.generate_clauses_labeled(
                word=word, word_index=idx, positive=True
            )
        for idx, word in enumerate(negative_sample):
            self.classifier.generate_clauses_labeled(
                word=word, word_index=idx, positive=False
            )
        self.model.setParam("MIPFocus", 1)

    def add_unlabeled_sample_constraints(
        self,
        sample: list[tuple[str, ...]],
        sample_freq: np.array = None,
        lower_bound: Optional[int] = None,
        upper_bound: Optional[int] = None,
    ):
        self.alpha = self.model.addMVar(
            shape=(len(sample), self.N), vtype=GRB.BINARY, name="alpha"
        )
        for idx, word in enumerate(sample):
            self.classifier.generate_clauses_unlabeled(
                word=word, word_index=idx, alpha=self.alpha
            )

        alpha_sum = sample_freq @ self.alpha @ np.ones(self.N)
        ########################################################################################################################
        #                                           Bound Constraints:
        ########################################################################################################################
        if lower_bound:
            self.model.addConstr(
                alpha_sum >= lower_bound,
                "lower_bound",
            )
        if upper_bound:
            self.model.addConstr(
                alpha_sum <= upper_bound,
                "upper_bound",
            )

        ########################################################################################################################
        #                                           Optimization intial objective:
        ########################################################################################################################
        if lower_bound and upper_bound:
            self.model.setParam("MIPFocus", 1)
            self.model.ModelSense = GRB.MINIMIZE
        elif lower_bound:
            self.model.setParam("MIPFocus", 2)
            self.model.setObjectiveN(expr=alpha_sum, index=0, priority=1)
            self.model.ModelSense = GRB.MINIMIZE
        elif upper_bound:
            self.model.setParam("MIPFocus", 2)
            self.model.setObjectiveN(expr=alpha_sum, weight=-1, index=0, priority=1)
            self.model.ModelSense = GRB.MINIMIZE

    def add_distance_sample_constraints(
        self,
        sample: list[tuple[str, ...]],
        sample_pair_freq_matrix: np.ndarray,
        distance_matrix: np.ndarray,
    ):
        S = len(sample)
        self.alpha = self.model.addMVar(
            shape=(S, self.N), vtype=GRB.BINARY, name="alpha"
        )
        for idx, word in enumerate(sample):
            self.classifier.generate_clauses_unlabeled(
                word=word, word_index=idx, alpha=self.alpha
            )
        # self.acceptance_vector = self.model.addMVar(
        #     shape=(S, 1), vtype=GRB.BINARY, name="gamma"
        # )
        # self.rejected_vector = self.model.addMVar(
        #     shape=(S, 1), vtype=GRB.BINARY, name="beta"
        # )
        # self.model.addConstrs(
        #     (
        #         self.acceptance_vector[i, j] == self.alpha[i, :] @ np.ones(self.N)
        #         for i in range(S)
        #         for j in range(self.N)
        #     ),
        #     name="gamma",
        # )
        # self.model.addConstrs(
        #     (
        #         self.rejected_vector[i, j] == 1 - self.alpha[i, :] @ np.ones(self.N)
        #         for i in range(S)
        #         for j in range(self.N)
        #     ),
        #     name="beta",
        # )

        # self.cross_class_mask = self.model.addMVar(
        #     shape=(S, 1), vtype=GRB.BINARY, name="gamma"
        # )
        # cross_distance_sum = (
        #     self.cross_class_mask * distance_matrix * sample_pair_freq_matrix
        # ).sum()

        # self.normal_class_mark = self.rejected_vector @ self.rejected_vector.transpose()
        # normal_distance_sum = (
        #     self.normal_class_mark * distance_matrix * sample_pair_freq_matrix
        # ).sum()

        # self.model.setObjective(
        #     expr=cross_distance_sum - normal_distance_sum, sense=GRB.MAXIMIZE
        # )
        # self.model.setParam("MIPFocus", 2)

        self.acceptance_vector = self.model.addMVar(
            shape=(S,), vtype=GRB.BINARY, name="gamma"
        )
        self.rejected_vector = self.model.addMVar(
            shape=(S,), vtype=GRB.BINARY, name="beta"
        )
        self.model.addConstr(
            (self.acceptance_vector == self.alpha @ np.ones(self.N)),
            name="gamma",
        )
        self.model.addConstr(
            (self.acceptance_vector == 1 - self.alpha @ np.ones(self.N)),
            name="beta",
        )

        self.cross_class_mask = self.model.addMVar(
            shape=(S, S), vtype=GRB.BINARY, name="cross_class_mask"
        )
        for i in range(S):
            for j in range(S):
                self.model.addConstr(
                    self.cross_class_mask[i, j] <= self.acceptance_vector[i]
                )
                self.model.addConstr(
                    self.cross_class_mask[i, j] <= self.rejected_vector[j]
                )
                self.model.addConstr(
                    self.cross_class_mask[i, j]
                    >= self.acceptance_vector[i] + self.rejected_vector[j] - 1
                )
        cross_distance_sum = (
            self.cross_class_mask * distance_matrix * sample_pair_freq_matrix
        ).sum()

        self.normal_class_mark = self.model.addMVar(
            shape=(S, S), vtype=GRB.BINARY, name="normal_class_mask"
        )
        for i in range(S):
            for j in range(S):
                self.model.addConstr(
                    self.normal_class_mark[i, j] <= self.rejected_vector[i]
                )
                self.model.addConstr(
                    self.normal_class_mark[i, j] <= self.rejected_vector[j]
                )
                self.model.addConstr(
                    self.normal_class_mark[i, j]
                    >= self.rejected_vector[i] + self.rejected_vector[j] - 1
                )
        normal_distance_sum = (
            self.normal_class_mark * distance_matrix * sample_pair_freq_matrix
        ).sum()

        self.model.setObjective(
            expr=cross_distance_sum - normal_distance_sum, sense=GRB.MAXIMIZE
        )
        self.model.setParam("MIPFocus", 2)

    def add_sink_state_penalty(self, lambda_s, sink_state_index=1):
        """Add penalty for transitions not leading to the sink state."""
        sink_state = self.states[sink_state_index]
        self.model.setObjectiveN(
            expr=quicksum(
                1 - self.transition[q, a, sink_state]
                for q in self.states
                for a, _ in enumerate(self.alphabet)
            ),
            index=1,
            weight=lambda_s,
            priority=0,
        )

    def add_self_loop_penalty(self, lambda_l):
        """Add penalty for transitions that are not self-loops."""
        self.model.setObjectiveN(
            expr=quicksum(
                self.transition[q, a, q0]
                for q in self.states
                for a, _ in enumerate(self.alphabet)
                for q0 in self.states
                if q0 != q
            ),
            index=2,
            weight=lambda_l,
            priority=0,
        )

    def add_parallel_edge_penalty(self, lambda_p):
        """Add penalty for multiple successor states."""

        eq = self.model.addVars(self.N, self.N, vtype=GRB.BINARY, name="eq")
        for q in self.states:
            for q0 in self.states:
                self.model.addLConstr(
                    eq[q, q0]
                    <= sum(
                        self.transition[q, a, q0] for a, _ in enumerate(self.alphabet)
                    )
                )
                for a, _ in enumerate(self.alphabet):
                    self.model.addLConstr(eq[q, q0] >= self.transition[q, a, q0])
        self.model.setObjectiveN(
            expr=quicksum(eq[q, q0] for q in self.states for q0 in self.states),
            index=3,
            weight=lambda_p,
            priority=0,
        )
