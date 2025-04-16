import logging
import sys
from typing import Optional
from gurobipy import GRB, Model, quicksum
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
        self.model: Model = Model(f"Hypothesis_{self.N}")

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
        alpha = self.model.addMVar(
            shape=(len(sample), self.N), vtype=GRB.BINARY, name="alpha"
        )
        for idx, word in enumerate(sample):
            self.classifier.generate_clauses_unlabeled(
                word=word, word_index=idx, alpha=alpha
            )

        alpha_sum = sample_freq @ alpha @ np.ones(self.N)
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

    def add_distance_sample_constraints_linear(
        self,
        sample: list[tuple[str, ...]],
        sample_pair_freq_matrix: np.ndarray,
        distance_matrix: np.ndarray,
    ):
        S = len(sample)
        alpha = self.model.addMVar(shape=(S, self.N), vtype=GRB.BINARY, name="alpha")
        for idx, word in enumerate(sample):
            self.classifier.generate_clauses_unlabeled(
                word=word, word_index=idx, alpha=alpha
            )
        acceptance_vector = alpha @ np.ones(self.N)

        # Create auxiliary variables: matrices of size n x n.
        # gamma[i,j] will indicate (alpha[i]==0 and alpha[j]==0).
        # beta[i,j] will indicate (alpha[i]==1 and alpha[j]==1).
        gamma = self.model.addMVar(shape=(S, S), vtype=GRB.BINARY, name="gamma")
        beta = self.model.addMVar(shape=(S, S), vtype=GRB.BINARY, name="beta")

        # Define phi (n x n) as the variable that will be related to gamma and beta.
        phi = self.model.addMVar(shape=(S, S), lb=-GRB.INFINITY, name="phi")

        # ----- Diagonal constraint -----
        # For i = j, we want phi[i,i] = 0.
        # The .diag() method selects the diagonal elements.
        self.model.addConstr(phi.diagonal() == 0, name="phi_diag")

        # ----- Off-diagonal constraints -----
        # We now add the constraints in a vectorized manner.
        # Using broadcasting, alpha[:, None] is an (n x 1) column vector
        # and alpha[None, :] is an (1 x n) row vector.
        #
        # For gamma: (which indicates when both alphas are 0)
        #   gamma[i,j] ≤ 1 − alpha[i]
        #   gamma[i,j] ≤ 1 − alpha[j]
        #   gamma[i,j] ≥ 1 − alpha[i] − alpha[j]
        self.model.addConstr(
            gamma <= 1 - acceptance_vector[:, None], name="gamma_upper1"
        )
        self.model.addConstr(
            gamma <= 1 - acceptance_vector[None, :], name="gamma_upper2"
        )
        self.model.addConstr(
            gamma >= 1 - acceptance_vector[:, None] - acceptance_vector[None, :],
            name="gamma_lower",
        )

        # For beta: (which indicates when both alphas are 1)
        #   beta[i,j] ≤ alpha[i]
        #   beta[i,j] ≤ alpha[j]
        #   beta[i,j] ≥ alpha[i] + alpha[j] − 1
        self.model.addConstr(beta <= acceptance_vector[:, None], name="beta_upper1")
        self.model.addConstr(beta <= acceptance_vector[None, :], name="beta_upper2")
        self.model.addConstr(
            beta >= acceptance_vector[:, None] + acceptance_vector[None, :] - 1,
            name="beta_lower",
        )

        # ----- Relate phi to gamma and beta -----
        # The constraint for phi is defined as:
        #   phi[i,j] = 1 − 2*gamma[i,j] − beta[i,j]
        non_diag_mask = ~np.eye(phi.shape[0], dtype=bool)
        self.model.addConstr(
            phi[non_diag_mask] == 1 - 2 * gamma[non_diag_mask] - beta[non_diag_mask],
            name="phi_def",
        )

        distance_sum = (phi * distance_matrix * sample_pair_freq_matrix).sum()

        self.model.setObjective(expr=distance_sum, sense=GRB.MAXIMIZE)
        self.model.setParam("MIPFocus", 2)

    def add_distance_sample_constraints_quadratic(
        self,
        sample: list[tuple[str, ...]],
        sample_pair_freq_matrix: np.ndarray,
        distance_matrix: np.ndarray,
    ):
        S = len(sample)
        alpha = self.model.addMVar(shape=(S, self.N), vtype=GRB.BINARY, name="alpha")
        for idx, word in enumerate(sample):
            self.classifier.generate_clauses_unlabeled(
                word=word, word_index=idx, alpha=alpha
            )
        acceptance_vector = self.model.addMVar(
            shape=(S,), vtype=GRB.BINARY, name="gamma"
        )
        rejected_vector = self.model.addMVar(shape=(S,), vtype=GRB.BINARY, name="beta")
        self.model.addConstr(
            (acceptance_vector == alpha @ np.ones(self.N)),
            name="gamma",
        )
        self.model.addConstr(
            (rejected_vector == 1 - alpha @ np.ones(self.N)),
            name="beta",
        )
        cross_class_mask = (
            acceptance_vector.reshape(-1, 1)
            @ rejected_vector.reshape(-1, 1).transpose()
        )
        cross_distance_sum = (
            cross_class_mask * distance_matrix * sample_pair_freq_matrix
        ).sum()

        normal_class_mark = (
            rejected_vector.reshape(-1, 1) @ rejected_vector.reshape(-1, 1).transpose()
        )
        normal_distance_sum = (
            normal_class_mark * distance_matrix * sample_pair_freq_matrix
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
