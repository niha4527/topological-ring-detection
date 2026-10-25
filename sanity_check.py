"""Does pyflagser behave as the project needs? Run: python sanity_check.py"""

import numpy as np
import pyflagser
from scipy.sparse import coo_matrix


def h1(edges, n=3):
    r, c = zip(*edges)
    M = coo_matrix(([0.1] * len(edges), (r, c)), shape=(n, n))
    return pyflagser.flagser_weighted(M, max_dimension=2, directed=True)["dgms"][1]


print("pyflagser", pyflagser.__version__)
print("cyclic     (expect [[0.1 inf]]):", h1([(0, 1), (1, 2), (2, 0)]))
print("transitive (expect []):         ", h1([(0, 1), (1, 2), (0, 2)]))
