import copy
import random
import os
import json

import pandas
from sklearn.tree import DecisionTreeClassifier

import utilities


class Classifier:
    def __init__(self, step_list, no_of_trials, save_path, dataset_directory, dataset_name,
                 classifier=None, param_grid=None, test_size=0.2, max_seed_value=100_000,
                 random_state=None, exclude=None):
        self.random_state = utilities.initialize_random_state(random_state)

        if exclude is None:
            exclude = ['y', 'steps', 'instance_id']
        if classifier is None:
            classifier = DecisionTreeClassifier
        if param_grid is None:
            param_grid = {'max_depth': 3}

        self.classifier = classifier
        self.param_grid = param_grid
        self.step_list = step_list
        self.save_path = save_path
        self.dataset_name = dataset_name
        self.dataset_directory = dataset_directory
        self.test_size = test_size
        self.exclude = exclude

        self.input_df = pandas.read_csv(os.path.join(self.dataset_directory, self.dataset_name))

        self.seed_list = [random.randint(0, max_seed_value) for _ in range(no_of_trials)]
        self.column_list = [i for i in sorted(self.input_df.columns) if i not in exclude]

        self.accuracy_list = None
        self.column_list = None

    def train_single_split(self, filtered_df, seed):
        r = utilities.stratified_train_test_split(filtered_df, exclude_columns=[], seed=seed,
                                                  test_size=self.test_size, column_list=self.column_list)
        x_train, y_train, x_test, y_test = r

        _param_grid = copy.deepcopy(self.param_grid)
        _param_grid['random_state'] = seed
        model = self.classifier(**_param_grid)
        return model.score(x_test, y_test)

    def train(self):
        self.accuracy_list = [self.train_single_split(self.input_df, s) for s in self.seed_list]
        self.save_results()

    def save_results(self):
        _file_path = os.path.join(self.save_path, f'{self.dataset_name}.json')
        open(_file_path, 'w').write(json.dumps({'accuracy_list': self.accuracy_list}))
