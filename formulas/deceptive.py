import numpy as np
import math
import random
import copy

from formulas.uniform_random import gen_uniform_random_clause
import utilities


class DeceptiveFormula:
    """
    Generate a deceptive formula F_1 with a hidden satisfying assignment A based on "Generating Hard Satisfiable
    Formulas by Hiding Solutions Deceptively", Haixia Jia, Cristopher Moore & Doug Strain, 2005.
    """
    def __init__(self, n=10_000, r=4.68, q=0.618, k=3, u_list=None, random_state=None):
        self.n = n
        self.m = math.ceil(n * r)
        self.q = q
        self.k = k
        self.random_state = utilities.initialize_random_state(random_state)

        if u_list is None:
            self.u_list = [1]
        else:
            self.u_list = sorted(u_list)

        self.assignment = self.gen_random_assignment()
        self.formula_1 = []
        self.formula_list = []
        self.acceptance_prob = self._cache_acceptance_probabilities()
        self.variable_name_permutation = self._gen_variable_permutation()

        self._var_list = list(range(1, n + 1))
        self._parity_list = [-1, 1] * k
        
        # Deprecated - no longer used 
        # self._idx = 0
        # self._idx_list = []
        # self.altered_clause_idx = None
        
        self.gen_base_formula()

    def gen_random_assignment(self):
        return [None] + np.random.choice([True, False], self.n).tolist()

    def calculate_t(self, clause, return_sum=True):
        """
        We use list comprehension to implement the following logic:

        t = 0
        for literal in clause:

            # Negative literal and its assignment is False in A.
            if literal < 0 and not A[literal]:
                t += 1

            # Positive literal and its assignment is True in A.
            elif literal > 0 and A[literal]:
                t += 1

            # A doesn't satisfy the literal.
            else:
                pass

        """
        literals_satisfied = [1 if (literal < 0 and not self.assignment[-literal])
                              else (1 if literal > 0 and self.assignment[literal] else 0)
                              for literal in clause]
        if return_sum:
            return sum(literals_satisfied)
        else:
            return literals_satisfied

    def gen_base_formula(self):
        while len(self.formula_1) < self.m:
            _clause = gen_uniform_random_clause(
                n=self.n,
                k=self.k,
                var_list=self._var_list,
                parity_list=self._parity_list
            )
            _t = self.calculate_t(_clause)
            if random.random() < self.acceptance_prob[_t]:
                self.formula_1.append(_clause)

    def gen_formulas_without_satisfying_assignment(self):
        """
        1. Shuffle the clauses of formula_1 to ensure any clause is equally likely to be chosen.
        2. Ensure that the first u clauses do not satisfy the chosen assignment.
        """
        _formula = copy.deepcopy(self.formula_1)
        random.shuffle(_formula)

        unsatisfied_clauses = 0
        for clause_idx in range(self.u_list[-1]):
            clause = _formula[clause_idx][:]
            literals_satisfied = self.calculate_t(clause, return_sum=False)

            for literal_idx, val in enumerate(literals_satisfied):
                if val == 1:
                    _formula[clause_idx][literal_idx] = -_formula[clause_idx][literal_idx]

            unsatisfied_clauses += 1
            if unsatisfied_clauses in self.u_list:
                _permuted_formula = self._permute_variable_names(copy.deepcopy(_formula))
                self.formula_list.append(_permuted_formula)

    # Helper functions.
    def _cache_acceptance_probabilities(self):
        return [0.0] + [self.q ** t for t in range(1, self.k + 1)]
    
    def _gen_variable_permutation(self):
        permutation = list(range(1, self.n + 1))
        random.shuffle(permutation)
        return permutation

    def _permute_variable_names(self, formula):
        permutation = [None] + self.variable_name_permutation[:]
        assert len(self.variable_name_permutation) == self.n
        permuted_formula = [[np.sign(i) * permutation[abs(i)] for i in clause] for clause in formula]
        return permuted_formula

    # Deprecated functions.
    # def add_expt_controls(self, formula):
    #     formula = self._permute_variable_names(formula)
    #     formula = self._permute_clauses(formula)
    #     return formula
    #
    # @staticmethod
    # def _permute_clauses(formula):
    #     random.shuffle(formula)
    #     return formula
    #
    # def _deprecated_record_clauses_with_t_equal_1(self, _t, _clause):
    #     """
    #     1. Record the index of clauses if A satisfies only 1 of their literals.
    #     2. We randomly pick 1 of these clauses to create the second formula s.t. A is not a satisfying assignment.
    #     """
    #     # If t = 1, then save the index of this clause.
    #     if _t == 1:
    #         self._idx_list.append(self._idx)
    # 
    #     # Increment clause counter.
    #     self._idx += 1
    # 
    # def _deprecated_gen_formula_without_satisfying_assignment(self):
    #     """
    #     1. Randomly pick a clause s.t. t = 1.
    #     2. Flip the polarity of the literal that A satisfies this ensures that A doesn't satisfy that clause.
    #     """
    #     self.altered_clause_idx = random.sample(self._idx_list, 1)[0]
    #     altered_clause = self.formula_1[self.altered_clause_idx]
    # 
    #     def pos(x): return x > 0 and self.assignment[x]
    #     def neg(x): return x < 0 and not self.assignment[-x]
    #     literal_idx = [(idx, l) for idx, l in enumerate(altered_clause) if pos(l) or neg(l)]
    #     if len(literal_idx) == 0:
    #         raise utilities.CustomError('We cannot find a literal that A satisfies. Something has gone wrong.')
    # 
    #     (idx, l) = literal_idx[0]
    #     self.formula_2 = copy.deepcopy(self.formula_1)
    #     self.formula_2[self.altered_clause_idx][idx] = -l
