#!/usr/bin/env python
# coding: utf-8

# Original paper: [http://www.charuaggarwal.net/outl.pdf](http://www.charuaggarwal.net/outl.pdf)
# The brute force algorithm.

import pandas as pd
import itertools
from collections import defaultdict

from evood.utils import sparsity_coeff, enumerate_subsets

class BruteForce:

    def __init__(self):
        self.score_df = pd.DataFrame()
        self.grid_ranges_dict = {}

    def fit(self, data, m, k, phi):
        """

        Brute force algorithm, as specified in the paper.

        Parameters:
            data: (np.array) the original data source.
            m: (int) number of solutions to return.
            k: (int) number of sub-dimensions to work on. k should be less than the number of columns in `data`.
            phi: (int) the ideal number of equi-depth ranges.

        Return:
            df_return: (pandas.DataFrame) a Pandas dataframe where each row contains the selected column, selected grid ranges, sparsity coefficient of the solution, and number of points covered by the solution.
            orig_grid_ranges_dict: (dict) a dictionary that keeps transformed grid ranges information for all columns in `data`.

        """
        assert (k <= data.shape[1]), "ValueError: number of sub-dimensions to look at is {}, exceed the original data dimension {}".format(k, data.shape[1])

        sparsity_coeff_dict = defaultdict(dict)
        perm_cols = enumerate_subsets([x for x in range(data.shape[1])], k)
        orig_grid_ranges_dict = {}

        print("Calculating sparsity coefficients.")
        for i in range(len(perm_cols)):
            grid_ranges = list(itertools.product([x for x in range(phi)], repeat=len(perm_cols[i])))
            for g in grid_ranges:
                sparsity_coeff_score, grid_ranges, num_of_records = sparsity_coeff(data, phi, perm_cols[i], g)
                sparsity_coeff_dict[tuple(perm_cols[i])][g] = {}
                sparsity_coeff_dict[tuple(perm_cols[i])][g]['sparsity_coeff_score'] = sparsity_coeff_score
                sparsity_coeff_dict[tuple(perm_cols[i])][g]['num_records'] = num_of_records
                orig_grid_ranges_dict.update(grid_ranges)

        # sort by values
        score_df = pd.DataFrame.from_dict({(i, j): sparsity_coeff_dict[i][j]
                                           for i in sparsity_coeff_dict.keys()
                                           for j in sparsity_coeff_dict[i].keys()},
                                          orient='index').reset_index()
        score_df.columns = ['columns_selected', 'grid_ranges', 'sparsity_coeff_score', 'num_records']
        score_df = score_df.sort_values(by=['sparsity_coeff_score'], axis=0)
        # only keep values that have more than 0 data points.
        df_return = score_df[score_df['num_records'] > 0].iloc[:m]
        df_return = df_return.reset_index(drop=True)

        self.score_df = df_return
        self.grid_ranges_dict = orig_grid_ranges_dict

        return self.score_df, self.grid_ranges_dict
