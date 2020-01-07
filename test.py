# Testing to see if the algorithms work.

import pandas as pd
import numpy as np
import itertools
from sklearn import datasets
from collections import defaultdict
import random
import copy

from models.brute_force import BruteForce
from models.evolutionary import Evolutionary

wine = datasets.load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
# print(X[:5,:])
# print(y[:5])
print("Shape of X: {}".format(X.shape))

# test brute force
clf = BruteForce()
res, range_dict = clf.fit(X[:, :8], 5, 3, 5)
print(res)

# test evolutionary
clf = Evolutionary()
grid_ranges_dict, best_sol = clf.fit(5, 3, 50, 0.4, 0.7, 10, X, 5)
print(grid_ranges_dict)
print(best_sol)
