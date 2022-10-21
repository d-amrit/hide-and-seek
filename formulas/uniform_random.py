import numpy as np
import random
import math

import utilities


def gen_uniform_random_clause(n, k=3, var_list=None, parity_list=None):
    """
    :param n: Number of variables.
    :param k: Number of literals.
    :param var_list: Contains a list of all variables in the formula. Usually [1, ..., n].
    :param parity_list: [-1, 1] * k. random.sample requires the population to be larger than the sample. We need to
                        sample k signs for each literal with equal probability.
    :return: A randomly generated clause with k literals generates as follows:
        1. Each clause is created by sampling _without_ replacement from the list of all variables.
        2. Each variable in the clause is multiplied by 1 or -1 with equal probability.
    """
    if var_list is None:
        var_list = list(range(1, n + 1))
    if parity_list is None:
        parity_list = [-1, 1] * k

    clause = random.sample(var_list, k)
    parity = random.sample(parity_list, k)
    return np.multiply(clause, parity).tolist()


def create_random_sat_instance(n, r=4.267, k=3, m=None, random_state=None):
    """
    Create a random k-CNF formula with clause density r = m/n.
    """
    utilities.initialize_random_state(random_state)

    # If m is None, use ratio.
    if m is None:
        m = math.ceil(n * r)

    # Variables are represented as integers and their sign indicates whether they are a positive or negative literal.
    literal_list = list(range(1, n + 1))
    parity_list = [-1, 1] * k

    # Randomly generate m clauses each with k literals.
    formula = [gen_uniform_random_clause(n, k, literal_list, parity_list) for _ in range(m)]
    return formula
