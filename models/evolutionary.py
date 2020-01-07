#!/usr/bin/env python
# coding: utf-8

# Original paper: [http://www.charuaggarwal.net/outl.pdf](http://www.charuaggarwal.net/outl.pdf)
# Evolutionary algorithm

import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import random
import copy

import utils


class Evolutionary:

    def __init__(self):
        self.grid_ranges_dict = {}
        self.best_set = []

    # selection
    def selection(self, S, phi, data):

        """

        Select a group of solutions given a popluation.

        Parameters:
            S: (list) a list that contains multiple solution strings (aka population).
            phi: (int) the ideal number of equi-depth ranges.
            data: (np.array) the original data source.

        Return:
           a new, selected popluation.

        """

        p = len(S)
        new_pop = []
        score_dict = defaultdict(dict)
        orig_grid_ranges_dict = {}

        # get sparsity coeff
        for string in S:
            selected_cols, selected_ranges = utils.str_to_col_grid_lists(string)
            score_dict[tuple(selected_cols)][tuple(selected_ranges)] = {}
            sparsity_coeff_score, grid_ranges, num_of_records = utils.sparsity_coeff(data, phi, selected_cols, selected_ranges)
            score_dict[tuple(selected_cols)][tuple(selected_ranges)]['sparsity_coeff_score'] = sparsity_coeff_score
            score_dict[tuple(selected_cols)][tuple(selected_ranges)]['num_records'] = num_of_records
            orig_grid_ranges_dict.update(grid_ranges)

        # sort results by neg sparsity coeff
        score_df = pd.DataFrame.from_dict({(i,j): score_dict[i][j]
                               for i in score_dict.keys()
                               for j in score_dict[i].keys()},
                           orient='index').reset_index()
        score_df.columns = ['columns_selected','grid_ranges','sparsity_coeff_score','num_records']
        score_df = score_df.sort_values(by=['sparsity_coeff_score'],axis=0)
        score_df = score_df.reset_index(drop=True)

        summ = sum(score_df.index.values)+p
        prob_offset = 0

        # need r(i)
        for index, row in score_df[::-1].iterrows():
            score_dict[row['columns_selected']][row['grid_ranges']]['rank'] = p - index # this is r(i)
            score_dict[row['columns_selected']][row['grid_ranges']]['prob'] = prob_offset + (score_dict[row['columns_selected']][row['grid_ranges']]['rank'] / summ)
            prob_offset = score_dict[row['columns_selected']][row['grid_ranges']]['prob']

        # determine who goes in or not: roulette wheel selection
        rand_val = random.random()

        for string in S:
            selected_cols, selected_ranges = utils.str_to_col_grid_lists(string)
            likelihood = score_dict[tuple(selected_cols)][tuple(selected_ranges)]['prob']

            if rand_val < likelihood:
                new_pop.append(string)

        return new_pop

    # crossover subroutine
    def recombine(self, s1, s2, data, phi):

        """

        Recombine two solutions and evaluate their children fitness using the sparsity coefficients.

        Parameters:
            s1: (str) a string representing one solution.
            s2: (str) a string representing another solution.
            data: (np.array) the original data source.
            phi: (int) the ideal number of equi-depth ranges.

        Return:
            Two children strings of s1 and s2.

        """

        assert (len(s1) == len(s2)), "ERROR: lengths of the two input strings do not match!"

        # get the positions
        curr_pos = 0
        Q, R = set(), set()

        parent_dim = len(s1) - s1.count("*")
        kid1 = list(s1)
        best_score = np.inf
        best_kid = ""

        while curr_pos < len(s1):
            if s1[curr_pos] != "*" and s2[curr_pos] != "*":
                R.add(curr_pos) # set of positions in which neither s1 nor s2 is *
            elif s1[curr_pos] == "*" and s2[curr_pos] == "*":
                curr_pos += 1
                continue
            else:
                Q.add(curr_pos) # set of positions in which either s1 or s2 is *

            curr_pos += 1

        Q = list(Q)
        R = list(R)

        # manipulating R first: get all possible combinations of candidate strings & compare.
        # TODO: the logic here is a bit vague. Not the exact implementation like the paper describes.
        if len(R) > 0:
            potential_children = utils.get_all_possible_combinations(s1, s2, R)

            for c in potential_children:
                sparsity_coefficent = utils.get_sparsity_coeff_from_str(c, data, phi)
                if sparsity_coefficent < best_score:
                    best_kid = c
                    best_score = sparsity_coefficent

            kid1 = list(best_kid)

        # look into Q now
        if len(Q) > 0:
            for pos in Q:
                non_star_pos_val = s1[pos] if s1[pos] != "*" else s2[pos]
                tmp_str_list = copy.deepcopy(kid1)
                tmp_str_list = list(tmp_str_list) # otherwise will have weird errors
                tmp_str_list[pos] = non_star_pos_val
                new_sparsity_coeff = utils.get_sparsity_coeff_from_str("".join(tmp_str_list), data, phi)
                if new_sparsity_coeff < best_score and len(tmp_str_list) - tmp_str_list.count("*") < parent_dim: # cannot exchange if exceed parent dim
                    kid1 = "".join(tmp_str_list)
                    best_score = new_sparsity_coeff

        # get the 2nd kid by always picking the opposite from the 1st kid
        kid2 = ["*"]*len(s2)
        for col in range(len(s1)):
            if s1[col] == "*" and s2[col] == "*":
                continue
            else:
                kid2[col] = s2[col] if kid1[col] == s1[col] else s1[col]

        return "".join(kid1), "".join(kid2)

    def crossover(self, S, phi, data):

        """

        Perform crossover to a collection of candidates in a given population.

        Parameters:
            S: (list) a list that contains multiple solution strings (aka population).
            phi: (int) the ideal number of equi-depth ranges.
            data: (np.array) the original data source.

        Return:
            A new collection of candidate strings after crossover.

        """

        # randomly match solutions together, since they're supposed to have the same sub-dimensions already.
        random.shuffle(S)
        g1, g2 = [S[i::2] for i in range(2)]
        valid_range = min(len(g1), len(g2))
        results = []

        for index in range(valid_range):
            s1 = g1[index]
            s2 = g2[index]
            new_s1, new_s2 = self.recombine(s1, s2, data, phi)
            results.append(new_s1)
            results.append(new_s2)

        # handle the unmatched string; it has not gone through the crossover process.
        if len(g1) > len(g2):
            results.append(g1[-1])
        elif len(g1) < len(g2):
            results.append(g2[-1])

        return results

    def mutation(self, S, p1, p2, phi):

        """

        Mutation step of the evolutionary algorithm.

        Parameters:
            S: (list) a list that contains multiple solution strings (aka population).
            p1: (float) mutation probability for type I (positions in Q - positions that are *)
            p2: (float) mutation probability for type II (positions in R - positions that are NOT *)
            phi: (int) the ideal number of equi-depth ranges.

        Return:
            A new collection of candidate strings after mutation.

        """

        assert (p1 <= 1 and p1 >= 0), "ERROR: p1 should be within [0,1] as a probability!"
        assert (p2 <= 1 and p2 >= 0), "ERROR: p2 should be within [0,1] as a probability!"

        new_set = []

        for solution in S:

            # get the positions
            curr_pos = 0
            Q, R = [], []
            sol_list = list(solution)

            while curr_pos < len(solution):
                if solution[curr_pos] != "*":
                    R.append(curr_pos) # set of positions in s which are *
                else:
                    Q.append(curr_pos) # set of positions in s which are not *

                curr_pos += 1

            # determine whether to mutate in Q
            ran_choice = random.random()
            if ran_choice > p1:
                selected_Q = random.choice(Q)
                sol_list[selected_Q] = str(random.randint(0, phi - 1))
                Q.remove(selected_Q)
                R.append(selected_Q)

                selected_R = random.choice(R)
                sol_list[selected_R] = "*"
                R.remove(selected_R)
                Q.append(selected_R)

            # determine whether to mutate in R
            ran_choice = random.random()
            if ran_choice > p2:
                sol_list[random.choice(R)] = str(random.randint(0, phi - 1))

            new_set.append("".join(sol_list))

        return new_set

    # subroutine to generate a random population
    def initiate_seed_population(self, p, k, data, phi):

        """

        Randomly initialize a population to start with.

        Parameters:
            p: (int) the desired number of solutions to be included in the population.
            k: (int) number of sub-dimensions to work on. k should be less than the number of columns in `data`.
            data: (np.array) the original data source.
            phi: (int) the ideal number of equi-depth ranges.

        Return:
            A new collection of candidate strings after random initialization.

        """

        str_len = data.shape[1]
        population_set = []

        while len(population_set) < p:
            temp_str_list = ["*"]*str_len
            pos_sampled = random.sample(range(0, str_len - 1), k)
            val_sampled = random.sample(range(0, phi - 1), k)
            for i in range(len(pos_sampled)):
                temp_str_list[pos_sampled[i]] = str(val_sampled[i])

            population_set.append("".join(temp_str_list))

        return population_set

    # put all the subroutines together
    def fit(self, m, k, p, p1, p2, iterations, data, phi):

        """

        The main evolutionary algorithm.

        Parameters:
            m: (int) the desired of solutions to keep.
            k: (int) number of sub-dimensions to work on. k should be less than the number of columns in `data`.
            p: (int) the desired number of solutions to be included in the population.
            p1: (float) mutation probability for type I (positions in Q - positions that are *)
            p2: (float) mutation probability for type II (positions in R - positions that are NOT *)
            iterations: (int) number of iterations to run (use instead of the termination criterion)
            data: (np.array) the original data source.
            phi: (int) the ideal number of equi-depth ranges.

        Return:
            A dictionary containing the grid range information for all columns and the top `m` solutions.

        """

        S = self.initiate_seed_population(p, k, data, phi)
        best_set = pd.DataFrame()
        grid_ranges_dict = {}

        for i in range(iterations): # instead of using the 95% convergence idea, need to manually configure termination criterion
            S = self.selection(S, phi, data)
            S = self.crossover(S, phi, data)
            S = self.mutation(S, p1, p2, phi)

            # rank the solutions in S and get the top m results
            sparsity_coeff_dict = defaultdict(dict)
            orig_grid_ranges_dict = {}

            for sol in S:
                selected_cols, selected_ranges = utils.str_to_col_grid_lists(sol)
                sparsity_coeff_score, transformed_cat_info, nD = utils.sparsity_coeff(data, phi, selected_cols, selected_ranges)
                sparsity_coeff_dict[sol] = {}
                sparsity_coeff_dict[sol]['sparsity_coeff_score'] = sparsity_coeff_score
                sparsity_coeff_dict[sol]['num_records'] = nD
                orig_grid_ranges_dict.update(transformed_cat_info)

            sparsity_coeff_dict2 = {}
            index = 0
            for k in sparsity_coeff_dict:
                sparsity_coeff_dict2[index] = {}
                sparsity_coeff_dict2[index][k] = sparsity_coeff_dict[k]
                index += 1

            # sort by values
            score_df = pd.DataFrame.from_dict({(i,j): sparsity_coeff_dict2[i][j]
                               for i in sparsity_coeff_dict2.keys()
                               for j in sparsity_coeff_dict2[i].keys()},
                           orient='index').reset_index()

            score_df.columns = ['level_0', 'solution','sparsity_coeff_score','num_records']
            score_df = score_df.sort_values(by=['sparsity_coeff_score'],axis=0)
            # only keep values that have more than 0 data point.
            df_return = score_df[score_df['num_records'] > 0].iloc[:m]
            df_return = df_return.reset_index(drop=True)

            grid_ranges_dict.update(orig_grid_ranges_dict)
            best_set = best_set.append(df_return, ignore_index=True)

        best_set = best_set[:m]

        self.grid_ranges_dict = grid_ranges_dict
        self.best_set = best_set['solution'].tolist()

        return self.grid_ranges_dict, self.best_set
