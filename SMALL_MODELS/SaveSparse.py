#!/usr/bin/python

import numpy as np
import scipy

A = scipy.sparse.eye(10, format='csr')*np.pi

scipy.sparse.save_npz("SparseMatrix.npz", A)


print('done.')


