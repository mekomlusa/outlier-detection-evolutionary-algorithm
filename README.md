# outlier-detection-evolutionary-algorithm

A simple implementation of [Outlier Detection for High Dimensional Data](http://www.charuaggarwal.net/outl.pdf) in Python.

## Usage

For details, refer to `example.py`. Clone the project into your local machine and import the scripts. Right now, your data must reside in the same folder as the scripts.

Algorithms implemented:

* The naive brute force algorithm
* The evolutionary algorithm

The input could be either Numpy arrays or Pandas dataframes.

## TODO

* ~~Support Pandas DataFrames~~
* Let users to decide initialized positions (now randomly initialized, may not converge to the best results)
* Refactor the codes
* Make the package available on PyPI
