import numpy as np
import random

from features.feature_generation import update_clause_metrics
import utilities


SUPPORTED_SOLVERS = ['walksat', 'schonnings', 'prob_sat_poly_make_and_break', 'prob_sat_poly_break_only', 'poly_ls']


def poly_ls(beta=-0.08, **kwargs):
    """
    From Local Search for Hard SAT Formulas: The Strength of the Polynomial Law, Liu, Papakonstantinou (2016).
    We use parameters given for 3-SAT and simplify the expression. After simplification, kappa is not used.
    """
    def f(break_value):
        return np.power(np.power(1 + break_value, 2) + beta, -1)

    clause = kwargs['clause']
    literal_dict = kwargs['literal_dict']
    assignment = kwargs['assignment']

    # Get break values for each literal.
    list_of_break_values = [utilities.calc_break_value(assignment, x, literal_dict) for x in clause]

    # Score each literal.
    literal_scores = [f(b) for b in list_of_break_values]
    
    return literal_scores


def prob_sat_poly_make_and_break(c_m=-0.8, c_b=3.1, epsilon=1, **kwargs):
    """
    From Choosing Probability Distributions for Stochastic Local Search and the Role of Make versus Break, Balint and
    Schöning (2012).
    """
    def f(make_value, break_value):
        return np.power(make_value, c_m) / np.power(epsilon + break_value, -c_b)
    
    clause = kwargs['clause']
    literal_dict = kwargs['literal_dict']
    assignment = kwargs['assignment']
    unsat_clauses = kwargs['unsat_clauses']

    # Get make and break values for each literal.
    list_of_make_values = [utilities.calc_make_value(x, unsat_clauses) for x in clause]
    list_of_break_values = [utilities.calc_break_value(assignment, x, literal_dict) for x in clause]

    # Score each literal.
    literal_scores = [f(list_of_make_values[i], list_of_break_values[i]) for i in range(len(clause))]

    return literal_scores


def prob_sat_poly_break_only(c_b=2.3, epsilon=1, **kwargs):
    """
    From Choosing Probability Distributions for Stochastic Local Search and the Role of Make versus Break, Balint and
    Schöning (2012).
    """
    def f(break_value):
        return np.power(epsilon + break_value, -c_b)
    
    clause = kwargs['clause']
    literal_dict = kwargs['literal_dict']
    assignment = kwargs['assignment']

    # Get break values for each literal.
    list_of_break_values = [utilities.calc_break_value(assignment, x, literal_dict) for x in clause]

    # Score each literal.
    literal_scores = [f(b) for b in list_of_break_values]

    return literal_scores


def prob_sat_pick_var(**kwargs):
    """
    1. Read input from kwargs. We are doing this to ensure that we can easily switch between different pick_var
       functions.
    2. For each literal in the clause, calculate its break value. [Make?]
    3. Score each literal based on its make/break value. [Take the scoring function as input?]
    4. Normalize scores
    """
    clause = kwargs['clause']
    scoring_fn = kwargs['scoring_fn']

    # Get unnormalized scores based on input scoring function.
    literal_scores = scoring_fn(**kwargs)

    # Normalize scores so that they form a probability distribution.
    literal_scores = literal_scores / sum(literal_scores)

    # Draw a literal based on the probability distribution.
    draw = np.random.multinomial(1, literal_scores)
    _var_index = np.where(draw == 1)[0][0]
    var = clause[_var_index]

    return var


def schonnings_pick_var(**kwargs):
    """
    Randomly return a literal in the chosen UNSAT clause.
    """
    clause = kwargs['clause']
    return random.sample(clause, 1)[0]


def walk_sat_pick_var(**kwargs):
    """
    1. Find the break value of all variables.
    2. Find the min_break_value.
    3. If any variable has break value = 0 then u.a.r. pick such a variable and flip it.
    4. Else:
        1. With probability wp, u.a.r. pick a variable in the clause
        2. With probability (1 - wp), u.a.r. pick from variables wit the lowest break value.
    """
    clause = kwargs['clause']
    assignment = kwargs['assignment']
    literal_dict = kwargs['literal_dict']
    wp = kwargs['wp']

    break_value_dict = utilities.calc_break_value_dict(clause, assignment, literal_dict)

    min_break_value = min(break_value_dict.values())
    min_break_list = [key for key, value in break_value_dict.items() if value == min_break_value]

    if min_break_value == 0:
        var = random.sample(min_break_list, 1)[0]
    else:
        if random.random() < wp:
            var = random.sample(clause, 1)[0]
        else:
            var = random.sample(min_break_list, 1)[0]

    return var


def algorithm_name_to_fn_mapping(algorithm_name):
    """
    Map algorithm name to (pick_var, scoring_fn) functions.
    """
    if algorithm_name == 'walksat':
        return walk_sat_pick_var, None

    elif algorithm_name == 'schonnings':
        return schonnings_pick_var, None

    elif algorithm_name == 'prob_sat_poly_make_and_break':
        return prob_sat_pick_var, prob_sat_poly_make_and_break

    elif algorithm_name == 'prob_sat_poly_break_only':
        return prob_sat_pick_var, prob_sat_poly_break_only

    elif algorithm_name == 'poly_ls':
        return prob_sat_pick_var, poly_ls

    else:
        raise utilities.CustomError("Valid inputs are {'walksat', 'schonnings', 'prob_sat_poly_make_and_break', "
                                    "'prob_sat_poly_break_only', 'poly_ls'}. Please check the input string and ensure"
                                    "a valid algorithm is given.")


class StochasticLocalSearchAlgorithm:
    def __init__(self, formula, n, max_steps, min_steps=0, max_retries=1, wp=0.48, assignment=None,
                 algorithm_name='walksat', random_state=None, logging=False, use_custom_logger=False, **kwargs):
        self.formula = formula
        self.n = n
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.wp = wp
        self.pick_var, self.scoring_fn = algorithm_name_to_fn_mapping(algorithm_name)
        self.random_state = utilities.initialize_random_state(random_state)
        self.kwargs = kwargs

        # Memory-driven improvements.
        self.literal_dict = self._gen_literal_dict()

        # Initialize WalkSAT run-related parameters.
        self.attempts = 0
        self.assignment, self.initial_assignment = self.get_initial_assignment(assignment)
        self.steps = 0
        self.sat_assignment_found = False
        self.unsat_clauses = None

        # Logging-related parameters.
        self.logging = logging
        self.use_custom_logger = use_custom_logger
        self.computation_history = []
        if logging:
            self.clause_indices, self.unsat_dict = self._gen_clause_indices()
            self.flip_history = []
            self.no_of_unsat_clauses = []
            self.hamming_distance = []

    def get_initial_assignment(self, assignment=None):
        """
        If start assignment is given use this else uniformly randomly pick a starting assignment.
        """
        if assignment is None:
            assignment = np.random.choice([True, False], self.n)
        return assignment, np.copy(assignment)

    def stopping_condition(self):
        """
        Returns true if satisfying assignment found or max_steps reached.
        """
        return self.sat_assignment_found or self.steps >= self.max_steps

    def pick_clause(self):
        return random.sample(self.unsat_clauses, 1)[0]

    def flip_var(self, var):
        variable_idx = abs(var) - 1
        self.assignment[variable_idx] = not self.assignment[variable_idx]

    def get_unsat_clauses(self, flipped_var=None):
        if self.unsat_clauses is None:
            self.unsat_clauses = [clause for clause in self.formula if
                                  not utilities.check_clause_satisfiability(np.copy(self.assignment), clause)]
        else:
            # Keep all UNSAT clauses that don't include the flipped var. Add clauses that are newly UNSAT.
            new_clauses = [tuple(clause) for clause in self.literal_dict[-1 * flipped_var]
                           if not utilities.check_clause_satisfiability(np.copy(self.assignment), clause)]
            self.unsat_clauses = [clause for clause in self.unsat_clauses if flipped_var not in clause] + new_clauses

        self.sat_assignment_found = len(self.unsat_clauses) == 0

    def log_computation(self, **kwargs):
        if self.logging:
            self.flip_history.append(kwargs['flipped_var'])
            self.no_of_unsat_clauses.append(len(self.unsat_clauses))
            self.hamming_distance.append((self.initial_assignment != self.assignment).sum())

            unsat_indices = [self.clause_indices[clause] for clause in self.unsat_clauses]
            for clause in unsat_indices:
                self.unsat_dict[clause] = update_clause_metrics(self.unsat_dict[clause], self.steps)

            self._custom_logger()

    def save_computation(self):
        self._end_ongoing_clause_walks()
        self.computation_history.append({
            'flip_history': self.flip_history,
            'no_of_unsat_clauses': self.no_of_unsat_clauses,
            'hamming_distance': self.hamming_distance,
            'unsat_dict': self.unsat_dict
        })

    def solve(self):
        # Repeat until max_retries reached.
        while self.attempts < self.max_retries and not self.sat_assignment_found:
            # Prepare for new attempt.
            if self.attempts > 0:
                self._reset_walksat_variables_before_retry()

            # Find the initial number of UNSAT clauses.
            self.get_unsat_clauses()

            # Main SLS algorithm.
            while not self.stopping_condition():
                clause = self.pick_clause()
                flipped_var = self.pick_var(**{
                    'clause': clause,
                    'assignment': self.assignment,
                    'literal_dict': self.literal_dict,
                    'wp': self.wp,
                    'scoring_fn': self.scoring_fn,
                    'unsat_clauses': self.unsat_clauses
                })
                self.flip_var(flipped_var)
                self.get_unsat_clauses(flipped_var=flipped_var)
                self.steps += 1

                self.log_computation(**{
                    'flipped_var': flipped_var,
                })

            if self.logging:
                self.save_computation()
            self.attempts += 1

    # Helper functions
    def _gen_clause_indices(self, unsat_dict=None):
        """
        Create a dictionary mapping clauses to indices. In each step, we will save indices instead of clauses. We do
        this to reduce the memory footprint.
        """
        if self.logging:
            unsat_dict = {}
            clause_indices = {}
            for index, clause in enumerate(self.formula):
                clause = tuple(clause)
                clause_indices[clause] = index
                unsat_dict[index] = {'#walks': 0, 'avg_walk': 0, 'walk_length': 0, 'last_idx': None}
        else:
            clause_indices = {clause: index for index, clause in enumerate(self.formula)}

        return clause_indices, unsat_dict

    def _gen_literal_dict(self):
        """
        What? We create a dictionary which maps literals to the clauses they are in.

        Purpose. At every step we flip the assignment of a literal. Let us say that at a given step, we make the
        literal x true. The only NEW clauses that are POSSIBLY satisfied are the ones that contain x. In a similar vein,
        the only NEW clauses that are POSSIBLY unsatisfied are the ones that contain -x. So instead of checking ALL
        clauses at every step, we check only these clauses to speed up the process.
        """
        literal_list = list(np.concatenate((
            np.arange(1, self.n + 1, dtype=int),
            np.arange(-self.n, 0, dtype=int)
        )))
        literal_dict = {literal: [] for literal in literal_list}
        for clause in self.formula:
            for literal in set(clause):
                literal_dict[literal].append(clause)
        return literal_dict

    def _reset_walksat_variables_before_retry(self):
        self.assignment, self.initial_assignment = self.get_initial_assignment()
        self.steps = 0
        self.sat_assignment_found = False
        self.unsat_clauses = None
        if self.logging:
            self.clause_indices, self.unsat_dict = self._gen_clause_indices()
            self.flip_history = []
            self.no_of_unsat_clauses = []
            self.hamming_distance = []

    def _end_ongoing_clause_walks(self):
        """
        For computational efficiency, we update length of clause walk only when a clause switched from SAT to UNSAT. If
        we are terminating the walk before it was switched, we need to close the walk.
        """
        for clause, _ in enumerate(self.formula):
            if self.unsat_dict[clause]['last_idx'] is not None:
                self.unsat_dict[clause] = update_clause_metrics(self.unsat_dict[clause], self.steps, unsat_flag=False)

    def _custom_logger(self):
        """
        If use_custom_logger is true then give the input kwargs AND formula computation (thus far) to the
        custom logger.
        """
        if self.use_custom_logger:
            self.kwargs['custom_logger'](**{
                'input_kwargs': self.kwargs,
                'formula_computation': self
            })
