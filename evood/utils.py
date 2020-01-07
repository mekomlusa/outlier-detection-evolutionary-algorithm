#!/usr/bin/env python
# coding: utf-8

# Original paper: [http://www.charuaggarwal.net/outl.pdf](http://www.charuaggarwal.net/outl.pdf)
# Utility functions.

import pandas as pd
import numpy as np
import itertools
from collections import defaultdict
import random
import copy


def sparsity_coeff(data, phi, selected_cols, selected_ranges):
    """

    Calculate sparsity coefficient.

    Parameters:
        data: (np.array) the original data source.
        phi: (int) the ideal number of equi-depth ranges.
        selected_cols: (list, tuple) index of columns selected from the data.
        selected_ranges: (list, tuple) index of grid ranges selected from the data. Range cannot exceed (phi - 1).

    Return:
        sparsity_coeff: (float) calculated sparsity coefficient.
        transformed_cat_info: (dict) transformed grid ranges for each selected column.
        nD: (int) number of data points covered given the selected columns and grid ranges.

    """
    # pre-checks
    assert (max(
        selected_ranges) < phi), "ValueError: max value in range is {}, exceeded the designated phi value {}!".format(
        max(selected_ranges), phi)
    assert (len(selected_cols) == len(
        selected_ranges)), "ShapeMismatch: Length of selected columns should match with selected grid ranges, but now there are {} columns and {} grid ranges.".format(
        len(selected_cols), len(selected_ranges))
    assert (max(selected_cols) < data.shape[
        1]), "ValueError: max column index is {}, exceeded the length of the data = {}!".format(max(selected_cols),
                                                                                                data.shape[1])

    f = (1 / phi)
    N = data.shape[0]
    k = len(selected_cols)
    transformed_cat_info = {}
    data_transformed = np.array([0] * N)

    for col in selected_cols:
        transformed_cat_info[col] = pd.qcut(data[:, col], phi).categories.to_tuples().to_list()
        new_coded = pd.qcut(data[:, col], phi, labels=False)
        data_transformed = np.vstack((data_transformed.T, new_coded)).T

    # get rid of the empty col at the beginning
    data_transformed = data_transformed[:, 1:]

    # construct condition statement to calculate nD
    condition_statement = ""
    for i in range(len(selected_ranges)):
        temp = "(data_transformed[:," + str(i) + "] == " + str(selected_ranges[i]) + ")"
        condition_statement += temp
        if i != len(selected_ranges) - 1:
            condition_statement += " & "

    nD = len(data_transformed[eval(condition_statement)])
    sparsity_score = (nD - N * f ** k) / np.sqrt(N * f ** k * (1 - f ** k))

    return sparsity_score, transformed_cat_info, nD


def enumerate_subsets(nums, upper=5, set_upper=True):
    """

    Get all subsets of an array (the empty set [] is excluded).

    Parameters:
        nums: (list) an array that needs to get subsets from.
        upper: (int, optional) maximum length of the generated subset to be included.
            If set_upper = False, subsets with lengths from 1 to `upper` will be included in the final result set.
            Otherwise, only subsets with length = `upper` will be included.
        set_upper: (boolean, optional) whether to return subsets that have length = `upper`.

    Return:
        a list of arrays that are the subsets of `nums`.

    """

    def backtrack(first=0, curr=[]):
        if len(curr) == k and len(curr) > 0:
            output.append(curr[:])
        for i in range(first, len(nums)):
            curr.append(nums[i])
            backtrack(i + 1, curr)
            curr.pop()

    output = []
    for k in range(upper + 1):
        backtrack()

    if set_upper:
        return_list = [x for x in output if len(x) == upper]
        return return_list
    else:
        return output

# def not optimized, but works!
def str_to_col_grid_lists(s):
    """

    Convert a string to selected columns and selected grid ranges.

    Parameters:
        s: (str) a string representing one solution.
            For instance, *3**9 means 2 out of 5 dimensions are selected; the second and the last columns are selected,
            and their corresponding grid ranges are 3 and 9. The function will return (1, 4) and (3, 9).

    Return:
        selected_cols (list): list of columns selected as indicated by the string.
        selected_ranges (list): list of grid ranges selected as indicated by the string.

    """

    selected_cols, selected_ranges = [], []

    for i in range(len(s)):
        if s[i] != "*":
            selected_cols.append(i)
            selected_ranges.append(int(s[i]))

    return selected_cols, selected_ranges


def col_grid_list_to_str(data, selected_cols, selected_ranges):
    """

    Convert selected columns and selected grid ranges to a string representation.

    Parameters:
        data: (np.array) the original data source.
        selected_cols (list): list of columns selected as indicated by the string.
        selected_ranges (list): list of grid ranges selected as indicated by the string.

        For instance, given the dataset with 5 columns and 184 records, with selected_cols = [1, 4] and selected_ranges = [3, 9],
        the function will return *3**9 ("*" represents "don't care what values are in that column").

    Return:
        a string representing one solution.

    """

    assert (max(selected_cols) < data.shape[
        1]), "ValueError: max column index is {}, exceeded the length of the data = {}!".format(max(selected_cols),
                                                                                                data.shape[1])

    str_list = ["*" for _ in range(data.shape[1])]
    for i in range(len(selected_cols)):
        str_list[selected_cols[i]] = str(selected_ranges[i])

    return "".join(str_list)


def get_sparsity_coeff_from_str(s, data, phi):
    """

    Get sparsity coefficient from a candidate string.

    Parameters:
        s: (str) a string representing one solution.
        data: (np.array) the original data source.
        phi: (int) the ideal number of equi-depth ranges.

    Return:
        The sparsity coefficient of the given solution.

    """

    selected_cols, selected_ranges = str_to_col_grid_lists(s)
    sparsity_coeff_score, _, _ = sparsity_coeff(data, phi, selected_cols, selected_ranges)
    return sparsity_coeff_score


def get_all_possible_combinations(s1, s2, R):
    """

    Enumerate all possible recombinations between two candidate strings.

    Parameters:
        s1: (str) a string representing one solution.
        s2: (str) a string representing another solution.
        R: (list) a list which stores the positions in s1 and s2 where neither of the characters is *.

    Return:
        candidates: (list) a list of all possible recombinations given the two candidate strings.

    """

    candidates = []

    permutation_sets = enumerate_subsets(R, set_upper=False)
    for positions in permutation_sets:
        s1_list = list(s1)
        s2_list = list(s2)

        for pos in positions:
            s1_orig = s1[pos]
            s2_orig = s2[pos]

            s1_list[pos] = s2_orig
            s2_list[pos] = s1_orig

        candidates.append("".join(s1_list))
        candidates.append("".join(s2_list))

    candidates = list(set(candidates))

    return candidates
