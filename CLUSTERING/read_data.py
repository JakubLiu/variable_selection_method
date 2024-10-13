#!/usr/bin/python

import numpy as np
path = '/media/DANE/DANE/covid19/SRA_container/mix99analyse/Solreg_mat.txt'

# read data
print('reading in data...')
"""
- 1st columns is SNP ID
- 2nd columns is value from mix99, which we will cluster on
"""
data = np.loadtxt(path, skiprows = 1, usecols = [2,3])
print('reading data done.')

# writing data
print('writing data...')
np.savetxt('/media/DANE/home/jliu/SRA/CLUSTERING/DataToCluster.txt', data,)
print('all done.')

"""
Output file format:

1.000000000000000000e+00 -3.805100000000000105e-08
2.000000000000000000e+00 -2.072100000000000032e-05
3.000000000000000000e+00 4.692500000000000389e-06
4.000000000000000000e+00 5.063400000000000196e-08
5.000000000000000000e+00 3.564800000000000263e-08
6.000000000000000000e+00 -3.012299999999999941e-06
7.000000000000000000e+00 -1.467599999999999933e-06
8.000000000000000000e+00 -7.986499999999999418e-08
9.000000000000000000e+00 -1.596500000000000088e-06
"""
