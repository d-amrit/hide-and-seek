import json
import os
from numpyencoder import NumpyEncoder
from joblib import Parallel, delayed
import random

import utilities
from formulas.deceptive import DeceptiveFormula
from formulas.uniform_random import create_random_sat_instance


if not os.path.exists('../data/formulas/'):
    os.makedirs('../data/formulas/')


class Formulas:
    def __init__(self, n, r_list, q_list, u_list, number_of_formulas, save_path, n_jobs=-1, create_unsat=True,
                 create_deceptive=True, create_pairs=True, random_state=None, k=3):
        self.random_state = utilities.initialize_random_state(random_state)

        self.n = n
        self.k = k
        self.r_list = r_list
        self.u_list = u_list
        self.q_list = q_list
        self.number_of_formulas = number_of_formulas
        self.create_unsat = create_unsat
        self.create_deceptive = create_deceptive
        self.create_pairs = create_pairs
        self.n_jobs = n_jobs

        self.save_path = self._create_directory(save_path)

    def create_single_formula(self, formula_idx, formula_type, r, q=None):
        random_state = random.randint(0, 10**9)
        formula_dict = {'random_state': random_state}

        if formula_type == 'unsat':
            file_name = self._gen_file_name(formula_idx, formula_type, r)
            file_path = os.path.join(self.save_path, file_name)
            if not os.path.exists(file_path):
                formula = create_random_sat_instance(
                    n=self.n,
                    r=r,
                    k=self.k,
                    random_state=random_state
                )
                self._save_formula(formula, formula_idx, formula_type, formula_dict, r)
            else:
                print(f'FILE EXISTS: {file_path}')
        else:
            file_name = self._gen_file_name(formula_idx, formula_type, r, q=q)
            file_path = os.path.join(self.save_path, file_name)
            if not os.path.exists(file_path):
                s = DeceptiveFormula(
                    n=self.n,
                    r=r,
                    q=q,
                    k=self.k,
                    u_list=self.u_list,
                    random_state=random_state
                )

                formula_dict['satisfying_assignment'] = [bool(i) for i in s.assignment[1:]]
                self._save_formula(s.formula_1, formula_idx, 'sat', formula_dict, r, q=q)
            
                if self.create_pairs:
                    s.gen_formulas_without_satisfying_assignment()
                    formula_dict['variable_name_permutation'] = s.variable_name_permutation
                    for idx, _formula in enumerate(s.formula_list):
                        self._save_formula(_formula, formula_idx, 'pair', formula_dict, r, q=q, u=s.u_list[idx])
            else:
                print(f'FILE EXISTS: {file_path}')
                
    def create_dataset_for_one_formula_type(self, formula_type, q=None):
        for r in self.r_list:
            Parallel(n_jobs=self.n_jobs)(delayed(self.create_single_formula)(
                formula_idx=f,
                formula_type=formula_type,
                r=r,
                q=q
            ) for f in range(1, self.number_of_formulas + 1))
        
    def create_dataset(self):
        if self.create_unsat:
            self.create_dataset_for_one_formula_type(formula_type='unsat')
        
        if self.create_deceptive:
            for q in self.q_list:
                self.create_dataset_for_one_formula_type(formula_type='deceptive', q=q)
    
    @staticmethod
    def _create_directory(save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return save_path

    def _gen_file_name(self, formula_idx, formula_type, r, q=None, u=None):
        prefix = f"{formula_type}_r_{r}_n_{self.n}"
        suffix = f"{'0' * (6 - len(str(formula_idx)))}{formula_idx}.json"
        middle = "_"
        if u is not None:
            middle = f"_u_{u}{middle}"
        if q is not None:
            middle = f"_q_{q}{middle}"
        return f"{prefix}{middle}{suffix}"

    def _save_formula(self, formula, formula_idx, formula_type, formula_dict, r, q=None, u=None):
        """
        :param formula: Formula to be saved.
        :param formula_idx: Formula index.
        :param formula_type: Type of formula.
        :param
    
        Eg: unsat_000001.json will have formula_idx = 1, formula_type = 'unsat'
        """
        file_name = self._gen_file_name(formula_idx, formula_type, r, q, u)
        file_path = os.path.join(self.save_path, file_name)
        formula_dict['formula'] = formula
        open(file_path, 'w').write(json.dumps(formula_dict, cls=NumpyEncoder))


if __name__ == '__main__':
    pass
