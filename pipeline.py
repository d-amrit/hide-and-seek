import os
import datetime

import utilities
from experiment.create_formulas import Formulas
from experiment.run_solver import SolversComputation
from experiment.create_dataset import Dataset
from experiment.train import Classifier
from solvers.sls_solvers import SUPPORTED_SOLVERS


class Paths:
    def __init__(self, working_directory):
        if working_directory is None:
            working_directory = 'data/'
        self.working_directory = working_directory    
        
        self.formulas = self.create_directory('formulas')
        self.solver = self.create_directory('solver_computation')
        self.datasets = self.create_directory('datasets')
        self.results = self.create_directory('results')
        self.figures = self.create_directory('figures')
    
    def create_directory(self, folder_name):
        _directory = f"{os.path.join(self.working_directory, folder_name)}/"
        if not os.path.exists(_directory):
            os.makedirs(_directory)
        return _directory
        

if __name__ == '__main__':
    args_dict = utilities.parse_arguments()
    utilities.initialize_random_state(args_dict['random_state'])
    paths = Paths(working_directory=args_dict['working_directory'])

    print(f"Started at {datetime.datetime.now().strftime('%H:%M:%S')}")
    if args_dict['create_formulas']:
        f = Formulas(
            n=args_dict['variables'],
            r_list=args_dict['r_list'],
            q_list=args_dict['q_list'],
            u_list=args_dict['u_list'],
            number_of_formulas=args_dict['number_of_formulas'],
            save_path=paths.formulas,
            create_unsat=args_dict['create_unsat'],
            create_deceptive=args_dict['create_deceptive'],
            create_pairs=args_dict['create_pairs'],
            n_jobs=args_dict['n_jobs'],
            random_state=args_dict['random_state'],
            k=args_dict['literals'],
        )
        f.create_dataset()
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} Created formulas")

    if args_dict['run_solver']:
        if args_dict['algorithm_name'] not in SUPPORTED_SOLVERS:
            raise utilities.CustomError(f"Please use a supported solver. The supported solvers are "
                                        f"{', '.join(SUPPORTED_SOLVERS)}.")

        s = SolversComputation(
            n=args_dict['variables'],
            step_list=args_dict['step_list'],
            no_of_trials=args_dict['no_of_trials'],
            save_path=paths.solver,
            formula_path=paths.formulas,
            algorithm_name=args_dict['algorithm_name'],
            wp=args_dict['noise_parameter'],
            random_state=args_dict['random_state'],
            delete_raw_files=args_dict['delete_raw_files'],
            architecture=args_dict['architecture'],
            n_jobs=args_dict['n_jobs'],
        )
        s.run()
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} Ran solver")

    if args_dict['create_dataset']:
        d = Dataset(
            n=args_dict['variables'],
            r_list=args_dict['r_list'],
            q_list=args_dict['q_list'],
            u_list=args_dict['u_list'],
            step_list=args_dict['step_list'],
            number_of_formulas=args_dict['number_of_formulas'],
            datasets_path=paths.datasets,
            solver_data_path=paths.solver,
            create_pairs=args_dict['create_pairs'],
            create_individual=args_dict['create_individual'],
            random_state=args_dict['random_state'],
            n_jobs=args_dict['n_jobs']
        )
        d.create()
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} Created dataset")

    if args_dict['train_model']:
        if args_dict['test_size'] <= 0 or args_dict['test_size'] >= 1:
            raise utilities.CustomError('Test size must be between 0 and 1.')

        for dataset_name in next(os.walk(paths.datasets))[2]:
            c = Classifier(
                step_list=args_dict['number_of_formulas'],
                no_of_trials=args_dict['classifier_no_of_trials'],
                save_path=paths.results,
                test_size=args_dict['test_size'],
                random_state=args_dict['random_state'],
                dataset_directory=paths.datasets,
                dataset_name=dataset_name
            )

    if args_dict['create_figures']:
        pass
