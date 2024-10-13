#!/usr/bin/python

import numpy as np
import scipy

file_in_sparse = '/media/DANE/DANE/covid19/SRA_container/SmallModelsOutput_Sparse_NonParallel.npz'
file_out_dense = '/media/DANE/DANE/covid19/SRA_container/SmallModelsOutput_Dense.txt'

print('loading sparse matrix into memory...')
MAT_Sparse = scipy.sparse.load_npz(file_in_sparse)
print('done.')

NumTotalRows = MAT_Sparse.shape[0]
NumCompletedRows = 0

print('converting sparse matrix to dense matrix and writing to file: ', file_out_dense, '...')
with open(file_out_dense, 'a') as f:

        for row in MAT_Sparse:  # convert each row of the sparse matrix to a dense matrix and write that row to a file

            current_row = row.toarray()[0]

            for value in current_row:
                f.write(str(value) + ',')

            f.write('\n')
            print("Processed ", NumCompletedRows/NumTotalRows * 100.00, "% of rows.")
            NumCompletedRows = NumCompletedRows + 1

print('all done.')

