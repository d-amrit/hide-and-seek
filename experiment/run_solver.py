import os
import random
import numpy as np
import pandas
import shutil
import json
from numpyencoder import NumpyEncoder
from joblib import Parallel, delayed

import utilities
from solvers.sls_solvers import StochasticLocalSearchAlgorithm
from features.feature_generation import feature_generation


class SolversComputation:
    def __init__(self, n, step_list, no_of_trials, save_path, formula_path, algorithm_name, wp=None,
                 random_state=None, max_seed_value=100_000, working_directory='data/', extension='json',
                 delete_raw_files=True, architecture='parallel', n_jobs=-1, clean_up=True):

        self.random_state = utilities.initialize_random_state(random_state)

        # Input variables
        self.n = n
        self.no_of_trials = no_of_trials
        self.algorithm_name = algorithm_name
        self.wp = wp
        self.working_directory = working_directory
        self.save_path = save_path
        self.formula_path = formula_path
        self.extension = extension
        self.delete_raw_files = delete_raw_files
        self.architecture = architecture
        self.n_jobs = n_jobs
        self.clean_up = clean_up

        # Calculated variables
        self.step_list = [int(i * n) for i in sorted(step_list)]
        self.seed_list = [random.randint(0, max_seed_value) for _ in range(no_of_trials)]
        self.expt_id = utilities.get_expt_id(n, algorithm_name, wp)
        self.wip_folder, self.partial_folder = self.get_folder_paths()

        self.all_formulas = set([i.replace(f'.{self.extension}', '') for i in next(os.walk(self.formula_path))[2]
                                 if f'.{self.extension}' in i])
        self.pending_formulas = self.get_pending_formulas()

    def get_folder_paths(self):
        """
        wip_folder = We copy formulas from formula_path here when we start working on them. The idea is to tell
                     other machines not to work on this formula because someone else has already begun working on it.
        partial_folder = This is where the partial files for a given are stored. We are separating this from the
                         completed folder to ensure that the completed folder only has ready-to-download files.
        """
        wip_folder = f'{self.working_directory}work_in_progress/'
        partial_folder = f'{self.working_directory}partial/'

        for folder_path in [wip_folder, partial_folder, self.save_path]:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        return wip_folder, partial_folder

    def get_pending_formulas(self):
        """
        1. Pending formulas = All - WIP - Completed.
        2. We need to do this because we are simultaneously running this code on multiple independent (uncoordinated)
           machines, therefore, at each step we need to update the state of this machine by checking the progress
           made by other machines.
        """
        wip_formulas = set([i.replace(f'.{self.extension}', '') for i in next(os.walk(self.wip_folder))[2]])
        completed_formulas = set([i.replace('.csv', '') for i in next(os.walk(self.save_path))[2]])
        pending_formulas = self.all_formulas.difference(wip_formulas)
        pending_formulas = pending_formulas.difference(completed_formulas)
        return list(pending_formulas)

    def sample_formula(self):
        """
        1. Randomly selecting a formula from the pending formula list.
        2. Read it.
        3. Copy formula to WIP list.
        """
        instance_id = random.sample(self.pending_formulas, 1)[0]
        formula = self._read_formula(instance_id)
        shutil.copy2(self.formula_path + f'{instance_id}.{self.extension}',
                     self.wip_folder + f'{instance_id}.{self.extension}')
        return instance_id, formula

    def run_experiment_for_one_formula(self, instance_id, formula):
        """
        1. For each formula, a complete run is {{TRIAL}} independent walks of length at most {{max_steps}}.
        2. max_steps is a function of the size of the formula.
        """
        max_steps = self.step_list[-1]
        for seed in self.seed_list:
            # Name of the path for the largest steps.
            file_path = self._get_file_path(instance_id, seed, max_steps)
            if self._check_if_this_setting_has_already_been_run(file_path):
                # Folder where we keep the snapshots of each file.
                folder_path = self._get_snapshot_folder(instance_id, seed)

                # Run algorithm
                walk_details, steps = self._run_algorithm(formula, max_steps, seed, folder_path)

                # Save feature dataframes
                self._save_feature_dataframes(walk_details, seed, steps, max_steps, instance_id, folder_path,
                                              file_path)

        self._concat_feature_dataframes(instance_id)
        self._mark_instance_as_completed(instance_id)

    def run(self):
        if self.architecture == 'distributed':
            self._run_distributed()
        elif self.architecture == 'parallel':
            self._run_parallel()
        else:
            raise utilities.CustomError("Please provide a supported architecture type. The supported architecture "
                                        "types are 'distributed' and 'parallel'.")

        if self.clean_up:
            shutil.rmtree(self.wip_folder)
            shutil.rmtree(self.partial_folder)

    # Helper functions
    def _run_distributed(self):
        """
        We ran our experiments on a distributed uncoordinated compute cluster. This meant we ran into the issue of
        collisions, i.e., different machines working on the same formula. To avoid this, we used the machine's
        unique ID as a seed generator to ensure the randomness of each machine was different. After every formula was
        complete, we set a small random cool-off period and then checked the work-in-progress folder to ensure we
        didn't select a formula another machine was already working on.
        """
        while len(self.pending_formulas) > 0:
            instance_id, formula = self.sample_formula()
            self.run_experiment_for_one_formula(instance_id, formula)
            self.pending_formulas = self.get_pending_formulas()
            self._re_seed()

    def _run_parallel(self):
        Parallel(n_jobs=self.n_jobs)(delayed(self._run_single)(instance_id=f) for f in self.pending_formulas)

    def _run_single(self, instance_id):
        formula = self._read_formula(instance_id)
        self.run_experiment_for_one_formula(instance_id, formula)

    def _read_formula(self, instance_id):
        formula_path = os.path.join(self.formula_path, f'{instance_id}.json')
        with open(formula_path) as f:
            formula = json.load(f)
        formula = [tuple(i) for i in formula['formula']]
        return formula

    def _run_algorithm(self, formula, max_steps, seed, folder_path):
        """
        Run solver and store computation during its run.
        """

        s = StochasticLocalSearchAlgorithm(
            formula=formula,
            n=self.n,
            max_steps=max_steps,
            wp=self.wp,
            algorithm_name=self.algorithm_name,
            random_state=seed,
            logging=True,
            use_custom_logger=True,
            **{
                'custom_logger': self._logger,
                'logger_kwargs': {
                    'folder_path': folder_path,
                }
            }
        )
        s.solve()
        return self._get_walk_details(s), s.steps

    def _gen_features(self, walk_details, seed, steps, max_steps):
        """
        Generate features.
        """
        feature_dict = feature_generation(walk_details=walk_details, n=self.n)
        feature_dict['seed'] = seed
        feature_dict['steps'] = steps
        feature_dict['max_steps'] = max_steps
        return feature_dict

    def _get_file_path(self, instance_id, seed, max_steps):
        """
        Get file path for the feature dataframe for a fixed seed and max_steps setting.
        """
        return os.path.join(self.partial_folder, f'id_{instance_id}_ms_{max_steps}_r_{seed}.csv')

    def _get_snapshot_folder(self, instance_id, seed):
        """
        Create temporary folder during algorithm's run to store the snapshots.
        """
        folder_path = os.path.join(self.partial_folder + f'{instance_id}_{seed}')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return folder_path

    def _concat_feature_dataframes(self, instance_id):
        """
        1. Get list of partial files.
        2. Concat and save them as one file.
        3. Delete partial files.
        """
        list_of_partial_files = [i for i in next(os.walk(self.partial_folder))[2] if f'id_{instance_id}' in i]
        df_list = [pandas.read_csv(os.path.join(self.partial_folder, i)) for i in list_of_partial_files]

        df = pandas.concat(df_list, ignore_index=True)

        column_list = [i for i in df.columns if i not in ['seed', 'steps', 'max_steps']]
        df['instance_id'] = instance_id
        df['y'] = utilities.generate_label(instance_id)

        agg_dict = {i: np.mean for i in column_list}
        df = df.groupby(['instance_id', 'y', 'steps']).agg(agg_dict).reset_index()

        file_path = os.path.join(self.save_path, f'{instance_id}.csv')
        df.to_csv(file_path, index=False)

        if self.delete_raw_files:
            for file_name in list_of_partial_files:
                file_path = os.path.join(self.partial_folder, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)

    def _mark_instance_as_completed(self, instance_id):
        """
        Move file from WIP folder to Completed folder.
        """
        file_path = os.path.join(self.wip_folder, f'{instance_id}.{self.extension}')
        if os.path.exists(file_path):
            os.remove(file_path)

    def _re_seed(self):
        """
        1. We independently run process on multiple machines.
        2. If each machine has the same random_state then they will process the same formula.
        3. We initially break this symmetry by taking JOB_ID (must be unique because it is the machine ID!) as the
           initial random_state.
        4. Since we want to ensure ALL formulas are run with the same set of seeds, after the first formula, each
           machine will have random_state = NO_OF_TRIALS - 1.
        5. We re-seed the process with the JOB_ID after every formula to re-break the symmetry.
        """
        if self.random_state is not None:
            random.seed(self.random_state)

    def _logger(self, **kwargs):
        """
        We are going to make one long walk and take snapshots of the walk at the steps included in the step_list.
        """
        formula_computation = kwargs['formula_computation']
        steps = formula_computation.steps

        if steps in self.step_list:
            folder_path = kwargs['input_kwargs']['logger_kwargs']['folder_path']

            walk_details = self._get_walk_details(formula_computation)
            file_path = os.path.join(folder_path, f'{steps}.json')
            open(file_path, 'w').write(json.dumps(walk_details, cls=NumpyEncoder))
    
    @staticmethod
    def _get_walk_details(formula_computation):
        return {
            'flip_history': formula_computation.flip_history,
            'no_of_unsat_clauses': formula_computation.no_of_unsat_clauses,
            'unsat_dict': formula_computation.unsat_dict,
            'hamming_distance': formula_computation.hamming_distance
        }

    @staticmethod
    def _check_if_this_setting_has_already_been_run(file_path):
        file_created = os.path.exists(file_path) and os.path.getsize(file_path) > 0
        return not file_created
    
    def _create_feature_df_for_snapshot(self, folder_path, instance_id, seed,
                                        steps):
        _json_path = os.path.join(folder_path, f'{steps}.json')
        if os.path.exists(_json_path):
            with open(_json_path) as f:
                walk_details = json.load(f)

            _csv_path = self._get_file_path(instance_id, seed, steps)
            feature_dict = self._gen_features(walk_details, seed, steps, steps)
            pandas.DataFrame([feature_dict]).to_csv(_csv_path, index=False)
            
    def _save_feature_dataframes(self, walk_details, seed, steps, max_steps,
                                 instance_id, folder_path, file_path):
        """
        Save feature dataframes for all steps in step_list.
        """
        # This is the feature dataframe for max_steps.
        feature_dict = self._gen_features(walk_details, seed, steps, max_steps)
        pandas.DataFrame([feature_dict]).to_csv(file_path, index=False)

        # Create feature dataframe for each snapshot.
        for steps in self.step_list:
            if steps != max_steps:
                self._create_feature_df_for_snapshot(folder_path, instance_id, seed, steps)

        # We have saved each snapshot as a feature dataframe. Delete snapshots.
        shutil.rmtree(folder_path)


if __name__ == "__main__":
    pass
