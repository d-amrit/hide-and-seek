import os
import json
import pandas
import numpy as np
from joblib import Parallel, delayed
import random
from sklearn.tree import DecisionTreeClassifier

from pipeline import Paths
from experiment.create_formulas import Formulas
from features import feature_generation
from replicate_prior_results.predicting_e2e import read_formula_from_file
from utilities import stratified_train_test_split


SAVE_PATH = '/Users/amrit/Documents/Code/hide-and-seek/data/results/'
FORMULA_FOLDER = '/Users/amrit/Documents/Code/hide-and-seek/data/formulas/'
PREDICTING_E2E_DATA = '/Users/amrit/Downloads/data-2/600/'


def find_features(n, m, formula_name):
    # Read formula.
    if '.json' in formula_name:
        formula_path = os.path.join(FORMULA_FOLDER, formula_name)
        with open(formula_path) as f:
            formula = json.load(f)['formula']
    elif '.npz' in formula_name:
        formula_path = os.path.join(PREDICTING_E2E_DATA, formula_name)
        formula, _ = read_formula_from_file(formula_path, m)
    else:
        return

    # Generate features.
    feature_dict = feature_generation.calc_pos_neg_ratio_var_mean(formula, n, {})
    feature_dict = feature_generation.calculate_lp_slack_coeff_of_var(formula, n, feature_dict)

    # Save results.
    feature_dict_path = os.path.join(SAVE_PATH, formula_name.replace('.npz', '.json'))
    open(feature_dict_path, 'w').write(json.dumps(feature_dict))


def generate_data(n=600, m=2555, n_jobs=-1, number_of_formulas=5000, random_state=0,
                  working_directory=None):
    q_list = [0.2, 0.618]
    paths = Paths(working_directory=working_directory)
    f = Formulas(
        n=n,
        r_list=[4.258],
        q_list=q_list,
        u_list=[],
        number_of_formulas=number_of_formulas,
        save_path=paths.formulas,
        create_unsat=False,
        create_deceptive=True,
        create_pairs=False,
        n_jobs=n_jobs,
        random_state=random_state,
        k=3,
    )
    f.create_dataset()

    file_list = next(os.walk(FORMULA_FOLDER))[2] + next(os.walk(PREDICTING_E2E_DATA))[2]
    Parallel(n_jobs=n_jobs)(delayed(find_features)(
        n=n,
        m=m,
        formula_name=f
    ) for f in file_list)


def train(random_state=0, test_size=0.4, no_of_trials=25):
    random.seed(random_state)
    seed_list = random.sample(range(0, 100_000_000), no_of_trials)
    exclude_columns = ['instance_id', 'y']

    df_list = []
    for file_name in sorted(next(os.walk(SAVE_PATH))[2]):
        feature_dict_path = os.path.join(SAVE_PATH, file_name)
        with open(feature_dict_path) as f:
            _obj = json.load(f)

        _obj['instance_id'] = file_name.replace('.json', '')
        _df = pandas.DataFrame([_obj])
        df_list.append(_df)

    df = pandas.concat(df_list, ignore_index=True)
    df['y'] = df['instance_id'].apply(lambda x: 0 if 'unsat_' in x else 1)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Different types of formulas.
    q_200_mask = (df['instance_id'].astype(str).str.contains('q_0.2'))
    q_618_mask = (df['instance_id'].astype(str).str.contains('q_0.618'))
    unsat_mask = (df['instance_id'].astype(str).str.contains('unsat_'))
    sat_mask = ~(q_200_mask | q_618_mask | unsat_mask)

    # CHANGE 1/2: Change sat_mask to q_200_mask, q_618_mask to train on a different distribution.
    filtered_df = df.loc[df[sat_mask | unsat_mask].index, ].reset_index(drop=True)
    accuracy_list = []
    for seed in seed_list:
        r = stratified_train_test_split(filtered_df.copy(), seed=seed, test_size=test_size)
        x_train, y_train, x_test, y_test = r

        #     # --------------------------------------------------------------------------------
        #     # Train and test on different distributions
        #     # --------------------------------------------------------------------------------
        column_list = sorted([i for i in x_train.columns if i not in exclude_columns])
        x_train = x_train.loc[x_train.index, column_list]

        #     # Test: Keep UNSAT formulas.
        #     _unsat_mask = (x_test['instance_id'].astype(str).str.contains('unsat_'))
        #     x_test = x_test.loc[x_test[_unsat_mask].index, ].reset_index(drop=True)

        # CHANGE 2/2: Change sat_mask to q_200_mask, q_618_mask to test on a different distribution.
        #     # Test: Get SAT formulas from a different distribution.
        #     sat_test = df.loc[df[sat_mask].index, ].reset_index(drop=True)
        #     sat_test = sat_test.sample(n=len(x_test), random_state=seed).reset_index(drop=True)

        #     # Test: Get SAT formulas from a different distribution.
        #     _df_list = [x_test, sat_test]
        #     x_test = pandas.concat(_df_list, ignore_index=True)
        #     y_test = x_test['y']
        x_test = x_test.loc[x_test.index, column_list]
        #     # --------------------------------------------------------------------------------

        model = DecisionTreeClassifier(max_depth=2, random_state=seed)
        model = model.fit(x_train, y_train)
        accuracy_list.append(model.score(x_test, y_test))
    print(f'Accuracy % = {np.mean(accuracy_list) * 100.0} ({np.std(accuracy_list) * 100.0})')


if __name__ == '__main__':
    generate_data()
