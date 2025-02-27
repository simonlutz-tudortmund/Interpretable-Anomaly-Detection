from gurobipy import Model, GRB


def add_transition_constraints(model, states, alphabet, transitions):
    """Add constraints ensuring a single transition for each state-symbol pair."""
    for q in states:
        for a in alphabet:
            model.addConstr(sum(transitions[q, a, q_prime] for q_prime in states) == 1)  # single_transition_{q}_{a}


def add_prefix_constraints(model, states, prefixes, x):
    """Add constraints ensuring unique state for each prefix."""
    for w in prefixes:
        model.addConstr(sum(x[w, q] for q in states) == 1)  # single_run_{w}


def add_initial_state_constraint(model, x, start_token):
    """Add a constraint specifying the initial state."""
    model.addConstr(x[(start_token,), 0] == 1, "initial_state")


def add_transition_follow_constraints(model, states, prefixes, alphabet, x, transitions):
    """Add constraints for transition relationships between prefixes."""

    constr_batch = []
    for w in prefixes:
        for q in states:
            for a in alphabet:
                wa = tuple(list(w) + [a])
                if wa not in prefixes:
                    continue

                for q_prime in states:
                    constr_batch.append((w, q, a, q_prime, wa))

    model.addConstrs((x[w, q] + transitions[q, a, q_prime] - 1 <= x[wa, q_prime] for w, q, a, q_prime, wa in constr_batch))
    model.addConstrs((transitions[q, a, q_prime] >= x[wa, q_prime] for w, q, a, q_prime, wa in constr_batch))


def add_alpha_constraints(model, states, sample, x, final_states):
    """Add constraints for prefix acceptance."""
    # "alpha"
    alpha = model.addVars(((s, q) for s in sample for q in states), vtype=GRB.BINARY)
    for w in sample:
        for q in states:
            model.addConstr(alpha[w, q] >= x[w, q] + final_states[q] - 1)  # accept_{w}_{q}_1
            model.addConstr(alpha[w, q] <= x[w, q])  # accept_{w}_{q}_2
            model.addConstr(alpha[w, q] <= final_states[q])  # accept_{w}_{q}_3
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
    accepted = model.addVars(sample, vtype=GRB.BINARY) # "accepted"
    for w in sample:
        model.addConstr(accepted[w] <= sum(alpha[w, q] for q in states))  # accept_{w}_bound_upper
        for q in states:
            model.addConstr(accepted[w] >= alpha[w, q])  # accept_{w}_bound_lower_{q}
    return accepted


def add_gamma_constraints(model, sample, accepted):
    """Add constraints for prefix acceptance.
    gamma[w, w_] == 1 iff w_ is accepted and w is not
    so gamma[w, w_] == 1 iff w is not an outlier and w_ is an outlier
    """
    gamma = model.addVars(((s, s_) for s in sample for s_ in sample), vtype=GRB.BINARY)  # gammas

    for w in sample:
        for w_ in sample:
            if w == w_:
                continue
            model.addConstr(gamma[w, w_] >= accepted[w_] - accepted[w])  # accepted_and_rejected_{w}_{w_}_1
            model.addConstr(gamma[w, w_] <= accepted[w_])  # accepted_and_rejected_{w}_{w_}_2
            model.addConstr(gamma[w, w_] <= 1 - accepted[w])  # accepted_and_rejected_{w}_{w_}_3

    return gamma


def add_betta_constraints(model, sample, accepted):
    """Add constraints for prefix acceptance."""
    betta = model.addVars(((s, s_) for s in sample for s_ in sample), vtype=GRB.BINARY)  # bettas

    for w in sample:
        for w_ in sample:
            model.addConstr(betta[w, w_] >= 1 - accepted[w] - accepted[w_])  # rejected_and_rejected_{w}_{w_}_1
            model.addConstr(betta[w, w_] <= 1 - accepted[w])  # rejected_and_rejected_{w}_{w_}_2
            model.addConstr(betta[w, w_] <= 1 - accepted[w_])  # rejected_and_rejected_{w}_{w_}_3

    return betta


########################################################################################################################
#                                         For penalising ininterpretable DFAs:
########################################################################################################################

def add_interpretability_constrates(model, states, alphabet, transitions, final_states, lambda_s, lambda_p):
    if lambda_s != 0:
        add_sink_state_constraints(model, states, alphabet, transitions, final_states)

    eq = None
    if lambda_p != 0:
        eq = add_parallel_edge_constraints(model, states, alphabet, transitions)
    return eq


def add_sink_state_constraints(model, states, alphabet, transitions, final_states, sink_state_index=1):
    """Ensure sink state has only self-loops."""
    sink_state = states[sink_state_index]
    model.addConstr(final_states[sink_state] == 0)
    for a in alphabet:
        model.addConstr(transitions[sink_state, a, sink_state] == 1)


def add_parallel_edge_constraints(model, states, alphabet, transitions):
    """Ensure parallel edges are correctly represented."""
    eq = model.addVars(states, states, vtype=GRB.BINARY, name="eq")
    for q in states:
        for q0 in states:
            model.addConstr(eq[q, q0] <= sum(transitions[q, a, q0] for a in alphabet))
            for a in alphabet:
                model.addConstr(eq[q, q0] >= transitions[q, a, q0])
    return eq
