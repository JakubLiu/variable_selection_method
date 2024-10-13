#!/usr/bin/python

"""
The idea is that we will first write the 0:n rows of the sparse matrix to a dense matrix and save that file.
Next we will zip that file.
Then we will convert the n+1:2n rows of the sparse matrix to a dense matrix and save that file.
Then we will zip that file and so on...
"""


import numpy as np
import scipy

print('loading sparse matrix...')
MAT_Sparse = scipy.sparse.load_npz("/media/DANE/DANE/covid19/SRA_container/SmallModelsOutput_Sparse_NonParallel.npz")
print('done.')

print(MAT_Sparse)


NumRows = MAT_Sparse.shape[0]
Chunk = int(NumRows/1000)
iter = 0

print('converting to dense matrix...')

with open('/media/DANE/DANE/covid19/SRA_container/DenseMatrix_NonParallel_Chunk.txt', 'a') as f:
    
    while iter <= 4:
        for row in MAT_Sparse[0:Chunk,:]:  # convert each row of the sparse matrix to a dense matrix and write that row to a file
                print(str(iter/Chunk*100.00) + '%')
                current_row = row.toarray()[0]

                for value in current_row:
                    f.write(str(value) + ',')

                f.write('\n')
                iter = iter + 1

print('DONE.')

