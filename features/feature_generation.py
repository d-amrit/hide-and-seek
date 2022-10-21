import numpy as np
from catch22 import catch22_all
from itertools import chain


DECILES = np.arange(0, 1.1, 0.1)
QUANTILES = np.arange(0, 1.1, 0.25)


def calc_pos_neg_ratio_var_mean(formula, n, formula_details):
    """
    From Predicting Satisfiability at the Phase Transition, Xu et al. (2012)
    """

    # Flatten clauses
    flattened_clauses = list(chain(*formula.clauses))

    # Create variable list. None added for easier indexing.
    var_list = [None] + [[0, 0] for _ in range(1, n + 1)]

    # Get positive and negative counts.
    for literal in flattened_clauses:
        if literal > 0:
            var_list[literal][0] += 1
        else:
            var_list[-1 * literal][1] += 1

    # Calculate ratio.
    pos_neg_ratio = [abs(0.5 - i[0] / (i[0] + i[1])) if (i[0] + i[1]) != 0 else 0 for i in var_list[1:]]
    formula_details['pos_neg_ratio_var_mean'] = (2 * sum(pos_neg_ratio)) / n

    return formula_details


def calculate_lp_slack_coeff_of_var(formula, n, formula_details):
    """
    From Predicting Satisfiability at the Phase Transition, Xu et al. (2012)
    """
    from pulp import LpVariable, LpProblem, LpMaximize
    # Initialize maximization problem.
    problem = LpProblem("SAT_Relaxation", LpMaximize)

    # Create variable list. None added for easier indexing.
    var_list = [None] + [LpVariable('v{}'.format(i), 0, 1) for i in range(1, n + 1)]

    # Calculate objective and add constraint for each clause.
    objective = 0
    for clause in formula.clauses:
        # Number of literals in the clause that evaluate to true.
        no_true_literals = sum([var_list[literal] if literal > 0 else 1 - var_list[abs(literal)] for literal in clause])

        # Add constraint for each clause.
        problem += no_true_literals >= 1

        # Add sum to objective
        objective += no_true_literals

    # Set objective of LP problem.
    problem += objective

    # Solve the LP problem
    problem.solve()

    # Use rounding to get an integer solution to the relaxed problem. - Could use this for "warm" start.
    # lp_assignment = [int(round(v.varValue)) for v in problem.variables()]

    # Calculate coefficient of variation.
    lp_slack = [min(v.varValue, 1 - v.varValue) for v in problem.variables()]
    sigma_star, mu_star = np.std(lp_slack), np.mean(lp_slack)
    coeff_variation = sigma_star / mu_star if mu_star != 0 else 0
    formula_details.update({'sigma_star': sigma_star, 'mu_star': mu_star, 'coeff_variation': coeff_variation})

    return formula_details


def gen_break_features(feature_dict, break_list, max_break=5):
    for break_value in range(0, max_break + 1):
        feature_dict['break_count_{}'.format(break_value)] = break_list.count(break_value)
    feature_dict['break_more_than_{}'.format(max_break)] = len([i for i in break_list if i > max_break])
    feature_dict = add_catch_22_features(feature_dict, break_list, 'break', max_derivative=2)

    return feature_dict


def find_average_period(position_index_list, decimal_places=1):
    if position_index_list:
        for i in range(len(position_index_list) - 1):
            position_index_list[i] = position_index_list[i + 1] - position_index_list[i]
        position_index_list = position_index_list[:-1]
        if position_index_list:
            return np.round(np.mean(position_index_list), decimal_places)
    return np.nan


def add_catch_22_features(feature_dict, time_series, feature_initial, max_derivative, no_of_features=22):
    derivative = 0
    while derivative <= max_derivative:
        # Calculate features
        catch22_features = catch22_all(np.diff(time_series, n=derivative))

        # Save features
        for idx in range(no_of_features):
            feature_name = '{}_D{}_{}'.format(feature_initial, derivative, catch22_features['names'][idx])
            feature_dict[feature_name] = catch22_features['values'][idx]

        # Increment derivative
        derivative += 1

    return feature_dict


def calc_summary_statistics(feature_dict, list_of_values, feature_identifier, statistics=None):
    """
    Initial, final, average, minimum, maximum, #zeros.
    """

    if statistics is not None:
        if 'initial' in statistics:
            feature_dict['{}_Initial'.format(feature_identifier)] = list_of_values[0]
        if 'final' in statistics:
            feature_dict['{}_Final'.format(feature_identifier)] = list_of_values[-1]
        if 'zero' in statistics:
            feature_dict['{}_Zeros'.format(feature_identifier)] = list_of_values.count(0)
        if 'null' in statistics:
            feature_dict['{}_Null'.format(feature_identifier)] = np.count_nonzero(np.isnan(list_of_values))

    # Check to see if all values are NaN.
    if all([i != i for i in list_of_values]):
        return feature_dict

    mu, sigma = np.nanmean(list_of_values), np.nanvar(list_of_values)
    feature_dict['{}_Max'.format(feature_identifier)] = np.nanmax(list_of_values)
    feature_dict['{}_Min'.format(feature_identifier)] = np.nanmin(list_of_values)
    feature_dict['{}_Avg'.format(feature_identifier)] = mu
    feature_dict['{}_Med'.format(feature_identifier)] = np.nanmedian(list_of_values)
    feature_dict['{}_Var'.format(feature_identifier)] = sigma
    cov = np.nan if mu == 0 else (sigma / mu)
    feature_dict['{}_CoV'.format(feature_identifier)] = cov

    # Calculate quartiles and deciles statistics.
    feature_dict = calc_quantile_statistics(feature_dict, list_of_values, QUANTILES, feature_identifier)
    feature_dict = calc_quantile_statistics(feature_dict, list_of_values, DECILES, feature_identifier)

    return feature_dict


def calc_quantile_statistics(feature_dict, list_of_values, quantile_groups, feature_identifier):
    # Calculate quantile values.
    quantile_values = np.nanquantile(list_of_values, quantile_groups)
    for idx, value in enumerate(quantile_groups):
        feature_name = '{}_Q{}_Value'.format(feature_identifier, round(value, 3))
        feature_dict[feature_name] = quantile_values[idx]

    # Calculate variance and inter-quantile range.
    for q_idx in range(len(quantile_groups) - 1):
        # Get lower and upper value of quantile.
        a, b = round(quantile_groups[q_idx], 2), round(quantile_groups[q_idx + 1], 2)
        v_a, v_b = quantile_values[q_idx], quantile_values[q_idx + 1]

        # Calculate name of features.
        identifier = '{}_Q{}_{}'.format(feature_identifier, a, b)
        var_name = '{}_Var'.format(identifier)
        iqr_name = '{}_IQR'.format(identifier)

        # Get relevant values.
        relevant_values = [i for i in list_of_values if v_a <= i < v_b and i == i]

        if relevant_values:
            feature_dict[var_name] = np.nanvar(relevant_values)
            feature_dict[iqr_name] = relevant_values[0] - relevant_values[-1]
        else:
            feature_dict[var_name] = np.nan
            feature_dict[iqr_name] = np.nan

    return feature_dict


def gen_period_and_count_features(feature_dict, data_dict, time_series, metric_list, feature_initial,
                                  add_catch22=True, max_derivative=0):
    # Add variance and range for each type of feature.
    for feature_type in metric_list:
        # Calculate feature identifier.
        feature_identifier = '{}_{}'.format(feature_initial, feature_type)

        # Get list of values.
        list_of_values = sorted(data_dict[feature_type])

        # Calculate summary statistics
        statistics = ['zero'] if 'count' in feature_type else ['null']
        feature_dict = calc_summary_statistics(feature_dict, list_of_values, feature_identifier, statistics=statistics)

    # Add catch 22 features.
    if add_catch22:
        feature_dict = add_catch_22_features(feature_dict, time_series, feature_initial, max_derivative)

    return feature_dict


def calc_avg_period_and_count(dict_of_values, metric_list):
    summary_dict = {metric: [] for metric in metric_list}
    for value in dict_of_values.values():
        for metric in metric_list:
            if 'period' in metric:
                summary_dict[metric].append(find_average_period(value[metric]))
            else:
                summary_dict[metric].append(value[metric])
    return summary_dict


def gen_variable_features(feature_dict, first_x_flips, n):
    # Calculate count and period of each variable.
    temp_var_dict = {i: {'period': [], 'count': 0} for i in range(1, n + 1)}
    for idx, var in enumerate(first_x_flips):
        temp_var_dict[abs(var)]['period'].append(idx)
        temp_var_dict[abs(var)]['count'] += 1

    # Rename variables by rank of #flips.
    ranked_list = [(0, 0)] + sorted([(value['count'], key) for key, value in temp_var_dict.items()], reverse=True)
    ranked_dict = {j[1]: i for i, j in enumerate(ranked_list)}

    # We have replaced literals by their transformed variables.
    first_x_flips = [ranked_dict[abs(i)] for i in first_x_flips]

    # Create var_dict in the form {variable: {count: value, avg_period: value}}
    var_dict = calc_avg_period_and_count(temp_var_dict, ['count', 'period'])

    # Calculate features from variables.
    metric_list = ['count', 'period']
    feature_dict = gen_period_and_count_features(feature_dict, var_dict, first_x_flips, metric_list, 'var')

    return feature_dict


def gen_clause_features(feature_dict, no_of_unsat_clauses, unsat_dict):
    # I am not calculating features that take time but don't really add value.
    # if calc_period_count:
    #     # Initialize clause dict.
    #     metric_list = ['period', 'count']
    #     clause_dict = {clause_idx: {'period': [], 'count': 0} for clause_idx in range(m)}
    #
    #     # Calculate actual values.
    #     for idx, clause in enumerate(clauses_chosen):
    #         clause_dict[clause]['period'].append(idx)
    #         clause_dict[clause]['count'] += 1
    #
    #     # Summarize values.
    #     clause_dict = calc_avg_period_and_count(clause_dict, metric_list)
    #
    #     # Calculate time-series features using transformed clauses
    #     # Calculate count of each clause - for loop over .count so that we iterate over the list once.
    #     clause_count = {i: 0 for i in range(m)}
    #     for var in clauses_chosen:
    #         clause_count[var] += 1
    #
    #     # Rename variables by rank of #flips.
    #     ranked_list = sorted([(value, key) for key, value in clause_count.items()], reverse=True)
    #     ranked_dict = {j[1]: i for i, j in enumerate(ranked_list)}
    #     clauses_chosen = [ranked_dict[i] for i in clauses_chosen]
    #     feature_dict = add_catch_22_features(feature_dict, clauses_chosen, 'transformed_clauses', 2)
    #
    #     # Calculate time-series features using #UNSAT clauses at each time step.
    #     feature_dict = gen_period_and_count_features(feature_dict, clause_dict, no_of_unsat_clauses, metric_list,
    #                                                  'clauses', max_derivative=2)

    # Add features about duration a clause was UNSAT for.
    no_of_walks, avg_walk, steps_unsat = [], [], []
    for key, value in unsat_dict.items():
        w, a = value['#walks'], value['avg_walk']
        s = round(w*a, 0)
        no_of_walks.append(w)
        avg_walk.append(a)
        steps_unsat.append(s)
    feature_dict = calc_summary_statistics(feature_dict, no_of_walks, '#walks', statistics=['zero'])
    feature_dict = calc_summary_statistics(feature_dict, avg_walk, 'avg_walk', statistics=['zero'])
    feature_dict = calc_summary_statistics(feature_dict, avg_walk, 'steps_unsat', statistics=['zero'])

    # Add features about #unsat clauses.
    feature_dict = calc_summary_statistics(feature_dict, no_of_unsat_clauses, 'unsat_clauses',
                                           statistics=['zero', 'initial', 'final'])

    return feature_dict


def gen_stepwise_unsat_clauses(walk_details, min_steps):
    unsat_clauses = [None] + [[] for _ in range(min_steps)]
    for clause, clause_info in walk_details['unsat_dict'].items():
        if 'step_list' in clause_info:
            step_list = clause_info['step_list']
            for i, v in enumerate(step_list):
                if i % 2 == 0 and v <= min_steps:
                    for step in range(v, step_list[i+1] + 1):
                        unsat_clauses[step].append(clause)
    return unsat_clauses[1:]


def update_clause_metrics(clause_dict, steps, unsat_flag=True):
    # We are still on the same walk, increment walk_length
    if (clause_dict['last_idx'] is None or clause_dict['last_idx'] == steps - 1) and unsat_flag:
        clause_dict['walk_length'] += 1
    else:
        # We are starting a new walk. Calculate avg_walk_length then start new walk.
        # If this is your first walk then avg_walk = walk_length
        if clause_dict['#walks'] == 0:
            clause_dict['avg_walk'] = clause_dict['walk_length']
        else:
            clause_dict['avg_walk'] *= clause_dict['#walks']
            clause_dict['avg_walk'] += clause_dict['walk_length']
            clause_dict['avg_walk'] = round(clause_dict['avg_walk'] / (clause_dict['#walks'] + 1), 3)

        clause_dict['#walks'] += 1
        clause_dict['walk_length'] = 1

    # Update last_idx
    clause_dict['last_idx'] = steps

    return clause_dict


def feature_generation(walk_details, n, feature_dict=None):
    """
    --------------------------------------------------------------------------------------------------------------------
    DECISIONS
    --------------------------------------------------------------------------------------------------------------------
    I. catch 22 features for clauses_chosen
    --------------------------------------------------------------------------------------------------------------------
    We are not calculating catch 22 features for clauses_chosen chosen as we could not see any meaningful way in which
    it is:

        (a) Informative. What information does it provide that isn't present in the other features?
        (b) Time-series. How do we format it as a 3-point time series?
    --------------------------------------------------------------------------------------------------------------------
    II. Removed # of times the same set of UNSAT clauses appeared.
    --------------------------------------------------------------------------------------------------------------------
    This seems too noisy, and I'm not sure what value it provides.
    --------------------------------------------------------------------------------------------------------------------

    :param walk_details: Data from random walk.
    :param n: Number of variables.
    # :param formula_details: Contains info about whether the formula satisfiable or not and #steps taken.
    :param feature_dict: Dictionary of features. Initialize to empty dict to make it easier down the road to test
    subset of features.
    # :param dataframe_path: Location where feature_df is saved.
    :return: Dataframe with features.
    """
    if feature_dict is None:
        feature_dict = {}

    # Generate features from variables.
    feature_dict = gen_variable_features(feature_dict, walk_details['flip_history'], n)

    # Calculate features from clauses.
    feature_dict = gen_clause_features(feature_dict, walk_details['no_of_unsat_clauses'], walk_details['unsat_dict'])

    # # Calculate features from breaks: (a) Counts and (b) Time-series.
    # if walk_details['breaks']:
    #     feature_dict = gen_break_features(feature_dict, walk_details['breaks'])

    # Calculate features from Hamming Distance
    feature_dict = calc_summary_statistics(feature_dict, walk_details['hamming_distance'], 'hamming')
    feature_dict = add_catch_22_features(feature_dict, walk_details['hamming_distance'], feature_initial='HTS',
                                         max_derivative=2)

    # Convert to dataframe.
    return feature_dict
