#!/usr/bin/python

import numpy as np
import statsmodels.api as sm
import scipy

file_in = "/media/DANE/home/jliu/SRA/MAF01/SRA_InputData_MAF01.txt"
file_out = "/media/DANE/DANE/covid19/SRA_container/SmallModelsOutput_Sparse_NonParallel.npz"
SNP_ID_file_out = '/media/DANE/home/jliu/SRA/ANNOTATE_DATA/SNP_ID_post_SmallModels_shuffled_NonParallel.txt'

print('reading genotype data...')
data = np.loadtxt(file_in, dtype = np.int32, delimiter = ',')
print('done.')

print('creating the dependent and independent variable matrices...')
Y = data[-1,:]  # create the vector of phenotypes
data = data[:-1,:]   # remove the phenotype row from the matrix
print('done.')


print('creating SNP_ID column...')
NumSNPs = data.shape[0]
SNP_IDs = np.zeros(NumSNPs, dtype = np.int32)

ID = 1

for i in range(0, NumSNPs):
    SNP_IDs[i] = ID
    ID = ID + 1

SNP_IDs = SNP_IDs.astype(np.int32)
print('done.')

print('concatenating genotype data and SNP ID column...')
data = np.concatenate((SNP_IDs.reshape(-1,1), data), axis=1)
print('done.')


NumShuffles = 1000
print('Shuffling rows of the dataset ', NumShuffles, ' times...')

for i in range(0,NumShuffles):
    np.random.shuffle(data)   # perform 1000 rounds of shuffling
    print('Shuffling', i/NumShuffles*100.00, '% done.')

print('random shuffling done.')

print('removing shuffled SNP IDs from shuffled genotype data...')
SNP_IDs_shuffled = data[:,0]
data = data[:,1:]
print('done.')

print('saving shuffled SNP ID file...')
np.savetxt(SNP_ID_file_out, SNP_IDs_shuffled.astype(np.int32), fmt = '%i', delimiter=",")
print('done.')

TotalNumSNP, TotalNumPat = data.shape

NumSNPsinEachModel = int(np.floor(TotalNumPat/10))  # this is how many SNPs will be in each small model
print('Number of SNPs in each model:', NumSNPsinEachModel)

N_iterations = int(np.floor(TotalNumSNP/NumSNPsinEachModel))  # number of iterations to be performed
print('Number of iterations: ', N_iterations)

print('Starting the iterations ____________________________________________________________________________________________')


big_model = scipy.sparse.csr_matrix((N_iterations, TotalNumSNP+1), dtype=np.float32)

Iteration = 0
column_counter = 0
row_counter = 0

while Iteration < N_iterations:
    X = data[NumSNPsinEachModel*Iteration:NumSNPsinEachModel*(Iteration+1),:]
    print(Iteration/N_iterations*100.00,'%')

    try:
        log_reg = sm.GLM(Y, X.T, family = sm.families.Binomial(link=sm.families.links.Logit())).fit()
        model_performance = log_reg.deviance  # model deviance for the current logistic regression
        SNP_effects = log_reg.params
        Iteration = Iteration + 1

    except np.linalg.LinAlgError:
        continue

    # write the SNP effects to the corresponding columns
    big_model[row_counter, column_counter:column_counter+NumSNPsinEachModel-1+1] = SNP_effects

    # write the model performance to the last column
    big_model[row_counter,-1] = model_performance

    column_counter = column_counter + NumSNPsinEachModel

    row_counter = row_counter + 1

print('saving file...')
scipy.sparse.save_npz(file_out, big_model)
print('all done.')

