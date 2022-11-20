# Hide and Seek: Scaling Machine Learning for Combinatorial Optimization via the Probabilistic Method

This repository contains code for the paper Hide and Seek: Scaling Machine Learning for Combinatorial Optimization via the Probabilistic Method by Dimitris Achlioptas, Amrit Daswaney, and Periklis Papakonstantinou.

## Pipeline

All results can be replicated by running `pipeline.py` with the requisite arguments. 
The pipeline has 4 stages. We describe each stage with their relevant arguments below:

### 1. Create formulas

1. If you only want to create formulas, add `-F true`.
2. In this stage, you can create 3 types of k-SAT formulas with clause density m/n:
   1. Uniformly random k-SAT formulas,
   2. q-deceptive satisfiable formulas, and
   3. A pair of q-deceptive satisfiable formula F and a formula that differs from F by u clauses.
3. If you are not interested in replicating our results but want to use our code to generate formulas, then please see the `formulas` folder for details. 
4. The relevant input arguments are:

```python
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
```

### 2. Run solver and get computation-based features

1. This stage requires the formulas to be already present. If the formulas are present, add `-S true` else `-F true -S true`.
2. In this stage, we run an {{SLS solver}} independently for {{trials}} and record its value at each step in {{step_list}}.
3. Please see the `solver` folder for the currently supported solvers. Alternatively, you can add your own solver and run the experiment with it.
4. The relevant input arguments are:

```python
parser.add_argument('-a', '--algorithm_name', type=str, default='walksat', help='Name of SLS solver to use.')
parser.add_argument('-w', '--noise_parameter', type=float, default=0.48, help='Noise parameter for WalkSAT')
parser.add_argument('-s', '--step_list', nargs="*", type=float, default=[int(2**i) for i in range(-4, 2)],
                    help='List of max steps the SLS algorithm takes. They are defined in MULTIPLES of n.')
parser.add_argument("-t", "--no_of_trials", type=int, default=16,
                    help="Number of independent trials to run the SLS solver.")
parser.add_argument('-del', '--delete_raw_files', type=bool, default=True,
                    help='Whether to delete files storing the raw computation of the SLS solver.')
```

### 3. Create dataset 

1. This stage requires the formulas and solver computation to be already present. If the formulas and computations are already present, add `-D true` else `-F true -S true -D true`.
2. In this stage, we flatten the features from the JSON format to a more traditional CSV format.

### 4. Train model

1. This stage requires the dataset to be already present. If the dataset is already present, add `-M true` else `-F true -S true -D true -M true`.
2. We implemented the following classifiers: Decision Tree, Random Forest, AdaBoost, Gradient Boosting, Histogram Gradient Boosting, Multi-layer Perceptron., XGBoost, Cat Boost, TabNet, LightGBM, auto-sklearn, and FLAML. We include only the results for the Decision Tree as it achieved similar performance (>0.1% difference in classification accuracy) and was easily interpretable.  

### x. Managing different stages of the pipeline using the follo

1. Experiment-level parameters are given below: 

```python
# Stage 1: Create formulas
parser.add_argument('-F', '--create_formulas', type=bool, default=True, help='Create dataset with formulas.')

# Stage 2: Run solver on formulas
parser.add_argument('-S', '--run_solver', type=bool, default=False, help='Run solver on the generated formulas.')

# Stage 3: Create dataset (csv) used for training and testing.
parser.add_argument('-D', '--create_dataset', type=bool, default=True,
                    help='Create datasets from solver computation.')

# Stage 4: Train model
parser.add_argument('-M', '--train_model', type=bool, default=False, help='Train model on the dataset.')

# Seed for replication.
parser.add_argument('-R', '--random_state', type=int, default=0, help='Input random state using to seed the '
                                                                      'randomness generator.')

# Architecture-related parameters.
parser.add_argument('-J', '--n_jobs', type=int, default=-1, help='Maximum number of concurrently running jobs.')
parser.add_argument('-A', '--architecture', type=str, default='parallel',
                    help='Type of architecture used - distributed compute cluster or laptop.')
parser.add_argument('-W', '--working_directory', type=str, default=None,
                    help='Directory where all files are saved.')
```

## Issues

If you face any issues, please feel free to reach out to me at amrit_daswaney@berkeley.edu and I will do my best to help.

