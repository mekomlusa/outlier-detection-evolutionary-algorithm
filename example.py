# Testing to see if the algorithms work.

import pandas as pd
import numpy as np
import itertools
from sklearn import datasets
from collections import defaultdict
import random
import copy

from evood.models.brute_force import BruteForce
from evood.models.evolutionary import Evolutionary

# test using the sklearn dataset
# wine = datasets.load_wine()
# X = wine.data
# y = wine.target
# feature_names = wine.feature_names
# # print(X[:5,:])
# # print(y[:5])
# print("Shape of X: {}".format(X.shape))

# test dataframes, using the heart disease data from UCI: https://www.kaggle.com/ronitf/heart-disease-uci
heart_df = pd.read_csv("heart.csv")
print(heart_df.head())
X = heart_df.drop(['target'], axis=1)
y = heart_df['target']
feature_names = X.columns.values.tolist()
print("# of records in X: {}".format(len(X)))

# test brute force
# clf = BruteForce()
# #res, range_dict = clf.fit(X[:, :8], 5, 3, 5)
# res, range_dict = clf.fit(X[['age','sex','cp','trestbps','chol']], 5, 3, 5) # for dataframe
# print(res)
# print(range_dict)

# test evolutionary
clf = Evolutionary(p1=0.4, p2=0.7)
grid_ranges_dict, best_sol = clf.fit(5, 3, 50, 10, X, 5)
print(grid_ranges_dict)
print(best_sol)
