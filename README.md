## Hide and Seek: Scaling Machine Learning for Combinatorial Optimization via the Probabilistic Method

This repository contains code for the paper Hide and Seek: Scaling Machine Learning for Combinatorial Optimization via the Probabilistic Method by Dimitris Achlioptas, Amrit Daswaney, and Periklis Papakonstantinou.

All results can be replicated by running `pipeline.py` with the requisite arguments. The `parse_arguments()` function is located in the `utilities.py` file. We replicate here for the reader's ease:

```python
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
    parser.add_argument('-S', '--run_solver', type=bool, default=True, help='Run solver on the generated formulas.')

    # Stage 3: Create dataset (csv) used for training and testing.
    parser.add_argument('-D', '--create_dataset', type=bool, default=True,
                        help='Create datasets from solver computation.')

    # Stage 4: Train model
    parser.add_argument('-M', '--train_model', type=bool, default=True, help='Train model on the dataset.')

    # Stage 5: Create figures
    parser.add_argument('-P', '--create_figures', type=bool, default=True, help='Create figures used in the paper.')

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
    parser.add_argument('-f', '--number_of_formulas', default=10, help='Number of formulas to generate.')

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
```

