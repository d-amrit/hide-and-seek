import numpy as np
import random
import argparse
import pandas


class CustomError(Exception):
    pass


def initialize_random_state(random_state):
    """
    Seed the generator for different libraries to get consistent results.
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    return random_state


def generate_label(instance_id):
    """
    Based on the instance_id, generate the label.
    """
    if 'unsat' in instance_id or 'pair' in instance_id:
        return 0
    else:
        return 1


def get_expt_id(n, algorithm_name, wp):
    """
    We are going to run this experiment for many configurations. We want to be able to uniquely identify them. We
    do this through an experiment ID.
    """
    expt_id = f'n_{n}_A_{algorithm_name}'
    if wp is not None:
        expt_id += f'_wp_{wp}'
    return expt_id


def check_clause_satisfiability(assignment, clause):
    """
    :param assignment: Current assignment.
    :param clause: Clause.
    :return: Boolean value that indicates whether the current assignment satisfies the given clause.
    """
    for var in clause:
        var_is_true = (var > 0 and assignment[var - 1]) or (var < 0 and not assignment[abs(var) - 1])
        if var_is_true:
            return True
    return False


def calc_make_value(literal, unsat_clauses):
    """
    Make value = Clauses that were UNSAT that are now SAT.
    Given the list of currently UNSAT clauses and the proposed literal that will be flipped, its make value is the
    number of currently UNSAT clauses that contain the literal.
    """
    return len([clause for clause in unsat_clauses if literal in clause])


def calc_break_value(assignment, literal, literal_dict):
    """
    Given an assignment and a literal, find the number of new UNSAT clauses if that literal is flipped.
    Note: literal_dict is for performance purposes. It gives you the clauses which have the flipped literal.
    """
    variable_idx = abs(literal) - 1
    new_assignment = np.copy(assignment)
    new_assignment[variable_idx] = not new_assignment[variable_idx]
    _clause_list = literal_dict[-1 * literal]
    new_clauses = [clause for clause in _clause_list if not check_clause_satisfiability(new_assignment[:], clause)]
    return len(new_clauses)


def calc_break_value_dict(clause, assignment, literal_dict):
    return {literal: calc_break_value(np.copy(assignment), literal, literal_dict) for literal in clause}


def create_balanced_datasets(input_df, column='y', class_name=0, subsample=1, random_state=0):
    # Split dataframe based on class.
    mask = (input_df[column] == class_name)
    type_1 = input_df.loc[input_df[mask].index, ].reset_index(drop=True)
    type_2 = input_df.loc[input_df[~mask].index, ].reset_index(drop=True)

    if len(type_1) > len(type_2):
        type_1 = type_1.sample(n=len(type_2), random_state=random_state).reset_index(drop=True)
    elif len(type_1) < len(type_2):
        type_2 = type_2.sample(n=len(type_1), random_state=random_state).reset_index(drop=True)
    df_list = [
        type_1.sample(frac=subsample, random_state=random_state).reset_index(drop=True),
        type_2.sample(frac=subsample, random_state=random_state).reset_index(drop=True)
    ]
    df = pandas.concat(df_list, ignore_index=True)
    if not df.empty:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def create_train_test_split(df, seed, test_size, y, row_id, label_id):
    """
    Filter df for input class and then split df into train-test based on test_size.
    """
    mask = (df[label_id] == y)
    class_df = df.loc[df[mask].index, ].reset_index(drop=True)

    test = class_df.sample(frac=test_size, random_state=seed).reset_index(drop=True)

    test_ids = test[row_id].tolist()
    train_mask = (~class_df[row_id].isin(test_ids))
    train = class_df.loc[class_df[train_mask].index, ].reset_index(drop=True)

    return train, test


def concat_and_return_ids(df1, df2, row_id):
    df_list = [df1, df2]
    df = pandas.concat(df_list, ignore_index=True)
    return df[row_id].tolist()


def stratified_split(df, seed, test_size, row_id='instance_id', label_id='y'):
    """
    Create a train and test that has the same class distribution.
    """
    train_0, test_0 = create_train_test_split(df, seed, test_size, 0, row_id, label_id)
    train_1, test_1 = create_train_test_split(df, seed, test_size, 1, row_id, label_id)

    train_ids = concat_and_return_ids(train_0, train_1, row_id)
    test_ids = concat_and_return_ids(test_0, test_1, row_id)

    return train_ids, test_ids


def split_train_test(input_df, train_ids, test_ids, column_list, additional_columns=None):
    if additional_columns is None:
        additional_columns = []

    mask = (input_df['instance_id'].isin(train_ids))
    train_df = input_df.loc[input_df[mask].index, ].reset_index(drop=True)
    x_train, y_train = train_df[column_list], train_df['y']

    mask = (input_df['instance_id'].isin(test_ids))
    test_df = input_df.loc[input_df[mask].index, ].reset_index(drop=True)
    x_test, y_test = test_df[additional_columns + column_list], test_df['y']
    return x_train, y_train, x_test, y_test


def stratified_train_test_split(input_df, exclude_columns=None, seed=0, test_size=0.2, column_list=None,
                                row_id='instance_id', label_id='y'):
    # Initialize default values for exclude_columns and column_list.
    if exclude_columns is None:
        exclude_columns = []
    if column_list is None:
        column_list = sorted([i for i in input_df.columns if i not in exclude_columns])

    # Get train and test IDs.
    id_and_label = input_df.loc[input_df.index, ['instance_id', 'y']].drop_duplicates().reset_index(drop=True)
    train_ids, test_ids = stratified_split(id_and_label, seed, test_size, row_id=row_id, label_id=label_id)

    # Shuffle dataframe.
    input_df = input_df.sample(frac=1, random_state=seed).fillna(0).reset_index(drop=True)

    return split_train_test(input_df, train_ids, test_ids, column_list)



def parse_arguments():
    """
    Allow arguments to be passed via the command line.
    """
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------------------------------------------------------------
    # I. Stages of the experiment
    # --------------------------------------------------------------------------------------------------------------
    # Stage 1: Create formulas
    parser.add_argument('-F', '--create_formulas', type=bool, default=True, help='Create dataset with formulas.')

    # Stage 2: Run solver on formulas
    parser.add_argument('-S', '--run_solver', type=bool, default=False, help='Run solver on the generated formulas.')

    # Stage 3: Create dataset (csv) used for training and testing.
    parser.add_argument('-D', '--create_dataset', type=bool, default=False,
                        help='Create datasets from solver computation.')

    # Stage 4: Train model
    parser.add_argument('-M', '--train_model', type=bool, default=False, help='Train model on the dataset.')

    # Stage 5: Create figures
    parser.add_argument('-P', '--create_figures', type=bool, default=False, help='Create figures used in the paper.')

    # Seed for replication.
    parser.add_argument('-R', '--random_state', type=int, default=0, help='Input random state using to seed the '
                                                                          'randomness generator.')

    # Architecture-related parameters.
    parser.add_argument('-J', '--n_jobs', type=int, default=-1, help='Maximum number of concurrently running jobs.')
    parser.add_argument('-A', '--architecture', type=str, default='parallel',
                        help='Type of architecture used - distributed compute cluster or laptop.')
    parser.add_argument('-W', '--working_directory', type=str, default=None,
                        help='Directory where all files are saved.')

    # --------------------------------------------------------------------------------------------------------------
    # II. Formula-related variables
    # --------------------------------------------------------------------------------------------------------------
    # 1. Formula size.
    parser.add_argument("-k", "--literals", type=int, default=3, help="Number of literals in a clause")
    parser.add_argument("-n", "--variables", type=int, default=10_000, help="Number of variables")
    parser.add_argument("-r", "--r_list", nargs="*", type=float, default=[5, 7, 9],
                        help="List of Clause density (m/n)")

    # 2. Formula type.
    parser.add_argument('-q', '--q_list', nargs="*", type=float, default=[0.3, 0.4, 0.5, 0.618],
                        help='List of q values to use when generating formulas.')
    parser.add_argument('-u', '--u_list', nargs="*", type=float, default=[int(2**i) for i in range(11)],
                        help='When creating deceptive formula, list of u values. u is the number of randomly chosen'
                             'clauses that the planted assignment does not satisfy.')

    # 3. Experiment-related variables
    parser.add_argument('-cu', '--create_unsat', type=bool, default=True, help='Create unsatisfiable formulas.')
    parser.add_argument('-cd', '--create_deceptive', type=bool, default=True, help='Create deceptive formulas.')
    parser.add_argument('-ci', '--create_individual', type=bool, default=True,
                        help='Create dataset of individual formulas.')
    parser.add_argument('-cp', '--create_pairs', type=bool, default=True,
                        help='Create paired instances for the deceptive formulas.')
    parser.add_argument('-f', '--number_of_formulas', type=int, default=10, help='Number of formulas to generate.')

    # --------------------------------------------------------------------------------------------------------------
    # III. Solver-related variables
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('-a', '--algorithm_name', type=str, default='walksat', help='Name of SLS solver to use.')
    parser.add_argument('-w', '--noise_parameter', type=float, default=0.48, help='Noise parameter for WalkSAT')
    parser.add_argument('-s', '--step_list', nargs="*", type=float, default=[int(2**i) for i in range(-4, 2)],
                        help='List of max steps the SLS algorithm takes. They are defined in MULTIPLES of n.')
    parser.add_argument("-t", "--no_of_trials", type=int, default=16,
                        help="Number of independent trials to run the SLS solver.")
    parser.add_argument('-del', '--delete_raw_files', type=bool, default=True,
                        help='Whether to delete files storing the raw computation of the SLS solver.')

    # --------------------------------------------------------------------------------------------------------------
    # IV. Solver-related variables
    # --------------------------------------------------------------------------------------------------------------
    parser.add_argument('-cn', '--classifier_no_of_trials', type=bool, default=True,
                        help='Whether to delete files storing the raw computation of the SLS solver.')
    parser.add_argument('-ts', '--test_size', type=float, default=0.2, help='Fraction to use as test set.')

    return vars(parser.parse_args())
