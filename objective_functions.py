from gurobipy import GRB
from distance_functions import find_max_dist


def set_bounded_objective(model, states, sample, alpha, lower_bound, upper_bound):
    """Set the objective function for the model.

    - model: Gurobi model.
    - sample: List of sequences in the sample.
    - lower_bound lower number of outliers
    - upper_bound upper number of outliers
    """
    if lower_bound and upper_bound:
        model.setObjective(1, GRB.MINIMIZE)
    elif lower_bound:
        model.setObjective(sum(alpha[w, q] for w in sample for q in states), GRB.MINIMIZE)
    elif upper_bound:
        model.setObjective(sum(alpha[w, q] for w in sample for q in states), GRB.MAXIMIZE)


def add_distance_based_objective(model,
                                 sample,
                                 accepted,
                                 gamma,
                                 betta,
                                 distance_matrix,
                                 reg_constant=0,
                                 outlier_weight=0.5):
    """
    Add an objective to a Gurobi model based on distances between sequences.

    Parameters:
    - model: Gurobi model.
    - sample: List of sequences in the sample.
    - accepted: Dictionary of accepted variables for each sequence index.
    - gamma: Dictionary of gamma variables for sequence pairs.
    - distance_matrix: Precomputed distance matrix (numpy array).
    - reg_constant: regularization of number of outliers
    - outlier_weight: coefficient for forcing model to find more or less outliers
    """
    n = len(sample)
    max_dist = find_max_dist(distance_matrix)

    objective = sum(
        (1 - outlier_weight) * betta[sample[i], sample[j]] * distance_matrix[i, j] +
        outlier_weight * gamma[sample[i], sample[j]] * (0 - distance_matrix[i, j])
        for i in range(n) for j in range(n)
    ) + sum(reg_constant * accepted[sample[i]] for i in range(n))

    model.setObjective(objective, GRB.MINIMIZE)