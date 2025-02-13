from gurobipy import Model, GRB

def add_transition_constraints(model, states, alphabet, delta):
    """Add constraints ensuring a single transition for each state-symbol pair."""
    for q in states:
        for a in alphabet:
            model.addConstr(sum(delta[q, a, q_prime] for q_prime in states) == 1, f"single_transition_{q}_{a}")


def add_prefix_constraints(model, states, prefixes, x):
    """Add constraints ensuring unique state for each prefix."""
    for w in prefixes:
        model.addConstr(sum(x[w, q] for q in states) == 1, f"single_run_{w}")


def add_initial_state_constraint(model, x, start_token):
    """Add a constraint specifying the initial state."""
    model.addConstr(x[(start_token,), 0] == 1, "initial_state")


def add_transition_follow_constraints(model, states, prefixes, alphabet, x, delta):
    """Add constraints for transition relationships between prefixes."""
    for w in prefixes:
        for q in states:
            for a in alphabet:
                wa = tuple(list(w) + [a])
                if wa in prefixes:
                    for q_prime in states:
                        model.addConstr(x[w, q] + delta[q, a, q_prime] - 1 <= x[wa, q_prime],
                                        f"run_{w}_{a}_{q}_{q_prime}_1")
                        model.addConstr(delta[q, a, q_prime] >= x[wa, q_prime], f"run_{w}_{a}_{q}_{q_prime}_2")


def add_alpha_constraints(model, states, sample, x, final_states):
    """Add constraints for prefix acceptance."""
    alpha = model.addVars(((s, q) for s in sample for q in states), vtype=GRB.BINARY, name="alpha")
    for w in sample:
        for q in states:
            model.addConstr(alpha[w, q] >= x[w, q] + final_states[q] - 1, f"accept_{w}_{q}_1")
            model.addConstr(alpha[w, q] <= x[w, q], f"accept_{w}_{q}_2")
            model.addConstr(alpha[w, q] <= final_states[q], f"accept_{w}_{q}_3")
    return alpha


def add_sample_constraints(model, states, sample, alpha, lower_bound, upper_bound):
    """Add constraints on the number of accepted words."""
    if lower_bound:
        model.addConstr(sum(alpha[w, q] for w in sample for q in states) >= lower_bound, "lower_bound")
    if upper_bound:
        model.addConstr(sum(alpha[w, q] for w in sample for q in states) <= upper_bound, "upper_bound")


########################################################################################################################
#                                           For distance based approach:
########################################################################################################################


def add_accepted_constraints(model, states, sample, alpha):
    """Add constraints on accepted"""
    accepted = model.addVars(sample, vtype=GRB.BINARY, name="accepted")
    for w in sample:
        model.addConstr(accepted[w] <= sum(alpha[w, q] for q in states), f"accept_{w}_bound_upper")
        for q in states:
            model.addConstr(accepted[w] >= alpha[w, q], f"accept_{w}_bound_lower_{q}")
    return accepted


def add_gamma_constraints(model, sample, accepted):
    """Add constraints for prefix acceptance.
    gamma[w, w_] == 1 iff w_ is accepted and w is not
    so gamma[w, w_] == 1 iff w is not an outlier and w_ is an outlier
    """
    gamma = model.addVars(((s, s_) for s in sample for s_ in sample), vtype=GRB.BINARY, name="gammas")

    for w in sample:
        for w_ in sample:
            if w == w_:
                continue
            model.addConstr(gamma[w, w_] >= accepted[w_] - accepted[w], f"accepted_and_rejected_{w}_{w_}_1")
            model.addConstr(gamma[w, w_] <= accepted[w_], f"accepted_and_rejected_{w}_{w_}_2")
            model.addConstr(gamma[w, w_] <= 1 - accepted[w], f"accepted_and_rejected_{w}_{w_}_3")

    return gamma


def add_betta_constraints(model, sample, accepted):
    """Add constraints for prefix acceptance."""
    betta = model.addVars(((s, s_) for s in sample for s_ in sample), vtype=GRB.BINARY, name="bettas")

    for w in sample:
        for w_ in sample:
            model.addConstr(betta[w, w_] >= 1 - accepted[w] - accepted[w_], f"rejected_and_rejected_{w}_{w_}_1")
            model.addConstr(betta[w, w_] <= 1 - accepted[w], f"rejected_and_rejected_{w}_{w_}_2")
            model.addConstr(betta[w, w_] <= 1 - accepted[w_], f"rejected_and_rejected_{w}_{w_}_3")

    return betta