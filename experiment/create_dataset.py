import os
import pandas
import random
from joblib import Parallel, delayed
import warnings

import utilities


class Dataset:
    def __init__(self, n, r_list, q_list, u_list, step_list, number_of_formulas, datasets_path, solver_data_path,
                 balance_data=True, create_pairs=True, create_individual=True, random_state=None, n_jobs=-1):

        self.random_state = utilities.initialize_random_state(random_state)

        self.n = n
        self.r_list = r_list
        self.q_list = q_list
        self.u_list = u_list
        self.step_list = sorted(step_list)
        self.number_of_formulas = number_of_formulas
        self.balance_data = balance_data
        self.create_pairs = create_pairs
        self.create_individual = create_individual
        self.datasets_path = self._create_directory(datasets_path)
        self.n_jobs = n_jobs

        # Get path where solver data is stored.
        self.solver_data_path = solver_data_path

        # Get indices for pair of formulas experiment
        self.y_0, self.y_1 = self._split_formulas_into_pairs()

    def create_individual_formulas_dataset(self, r, q):
        dataset_name = f'Individual_r_{r}_q_{q}'

        file_list = [f for f in next(os.walk(self.solver_data_path))[2] if
                     f'sat_r_{r}_q_{q}' in f or f'unsat_r_{r}' in f]
        df_list = [pandas.read_csv(self.solver_data_path + f) for f in file_list]

        self._concat_and_save(df_list, dataset_name)

    def create_pairs_of_formulas_dataset(self, u, r, q):
        dataset_name = f'Pair_r_{r}_q_{q}_u_{u}'

        y_0_list = [self._create_single_pair_of_instances(instance_id=i, y=0, u=u, r=r, q=q) for i in self.y_0]
        y_1_list = [self._create_single_pair_of_instances(instance_id=i, y=1, u=u, r=r, q=q) for i in self.y_1]

        self._concat_and_save(y_0_list + y_1_list, dataset_name)

    def create(self):
        if self.create_individual:
            _config_list = [{'r': r, 'q': q} for r in self.r_list for q in self.q_list]
            Parallel(n_jobs=self.n_jobs)(delayed(self.create_individual_formulas_dataset)(r=c['r'], q=c['q'])
                                         for c in _config_list)

        if self.create_pairs:
            _config_list = [{'u': u, 'r': r, 'q': q} for u in self.u_list for r in self.r_list for q in self.q_list]
            Parallel(n_jobs=self.n_jobs)(delayed(self.create_pairs_of_formulas_dataset)(u=c['u'], r=c['r'], q=c['q'])
                                         for c in _config_list)

    # Helper functions
    @staticmethod
    def _create_directory(datasets_path):
        if not os.path.exists(datasets_path):
            os.makedirs(datasets_path)
        return datasets_path

    def _get_solver_data_path(self, working_directory, solver_data_path, algorithm_name, wp):
        if solver_data_path is None:
            if algorithm_name is None or (algorithm_name == 'walksat' and wp is None):
                raise utilities.CustomError('Please specify a solver_data_path OR the algorithm_name. If you are '
                                            'using WalkSAT, please additionally specify the wp parameter')

            expt_id = utilities.get_expt_id(self.n, algorithm_name, wp)
            solver_data_path = f'{working_directory}Completed_{expt_id}/'
        return solver_data_path

    def _concat_and_save(self, df_list, dataset_name):
        df = pandas.concat(df_list, ignore_index=True)
        for steps in self.step_list:
            _step_df = df.loc[df[df['steps'] == steps].index, ].reset_index(drop=True)
            if not _step_df.empty:
                if self.balance_data:
                    _step_df = utilities.create_balanced_datasets(_step_df)
                _file_path = os.path.join(self.datasets_path, f'{dataset_name}_max_steps_{steps}.csv')
                _step_df.to_csv(_file_path, index=False)
            else:
                raise warnings.warn(f'Dataset with name {dataset_name} for steps = {steps} is empty. Please check.')

    def _split_formulas_into_pairs(self):
        """
        For all datasets, we want the same pairs to be chosen together to ensure like-for-like comparison across
        different splits.
        """
        id_list = ['0' * (6 - len(str(i + 1))) + str(i + 1) for i in range(self.number_of_formulas)]
        y_0 = random.sample(id_list, len(id_list) // 2)
        y_1 = [i for i in id_list if i not in y_0]
        return y_0, y_1

    def _create_single_pair_of_instances(self, instance_id, y, u, r, q):
        if y == 0:
            f1, f2 = 'sat', 'unsat'
        else:
            f1, f2 = 'unsat', 'sat'

        feature_df1 = self._process_single_df_of_pair(f_type=f1, prefix='F1', instance_id=instance_id, y=y,
                                                      u=u, r=r, q=q)
        feature_df2 = self._process_single_df_of_pair(f_type=f2, prefix='F2', instance_id=instance_id, y=y,
                                                      u=u, r=r, q=q)

        if feature_df1.empty or feature_df2.empty:
            warnings.warn(f"Pair for {instance_id} was not generated as the solver was able to find a satisfying "
                          f"assignment for one or more of the formulas. {f1} instance size = {len(feature_df1)}, "
                          f"{f2} instance size = {len(feature_df2)}.")
            return pandas.DataFrame()

        feature_df = pandas.merge(feature_df1, feature_df2, how='left', on=['instance_id', 'y', 'steps'])
        return feature_df

    def _process_single_df_of_pair(self, f_type, prefix, instance_id, y, u, r, q):
        # Read formula.
        if f_type == 'sat':
            _file_name = f'sat_r_{r}_q_{q}_{instance_id}.csv'
        else:
            _file_name = f'pair_r_{r}_q_{q}_u_{u}_{instance_id}.csv'
        file_path = os.path.join(self.solver_data_path, _file_name)
        _feature_df = pandas.read_csv(file_path)

        if sorted(_feature_df['steps'].unique()) == self.step_list:
            return pandas.DataFrame()

        # Rename columns except the ones we are merging on.
        exclude = ['instance_id', 'y', 'steps']
        column_list = [i for i in _feature_df.columns if i not in exclude]
        _feature_df = _feature_df.rename(columns={i: f'{prefix}_{i}' for i in column_list})

        # Ensure that values are correct for both pairs. ~ instance_id is probably unnecessary!
        _feature_df['y'] = y
        _feature_df['instance_id'] = instance_id

        return _feature_df
