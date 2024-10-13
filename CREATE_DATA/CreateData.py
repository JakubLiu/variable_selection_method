#!/usr/bin/python
"""
This script reads the plink files aswell as the phenotype files and creates the data used in the further steps.
"""

import pandas_plink
import numpy as np

print('reading PLINK files as dask array...')
#snp_info,sample_info,genotypes  = pandas_plink.read_plink('full_covid_data_plink')
genotypes  = pandas_plink.read_plink('/media/DANE/home/jliu/SRA/MAF01/plinkmaf01')[2]
print('reading PLINK files done.')

print('converitng dask array dtype to int8...')
genotypes = genotypes.astype(np.int8)
print('done.')


print('creating genotype matrix...')
genotype_mat = genotypes.compute()
print('genotype matrix created.')

print('reading recoded phenotype file...')
# reading phenotype file as array of dtype integer
phenotype_array = np.loadtxt('/media/DANE/home/jszyda/Kuba_Liu/CANONICAL/SRA/FULL_DATA/ResistPhenotypes_sorted_recoded.txt', dtype=int)
print('reading recoded phenotype file done.')

print('changing dtype of phenotype array to int8...')
phenotype_array = phenotype_array.astype(np.int8)
print('done.')

print('concatenating genotype matrix and phenotypes...')

# adding the phenotypes as last row to genotype matrix
full_data = np.vstack([genotype_mat, phenotype_array])
print('concatenation done.')

print('removing patients with undefined phenotypes...')
full_data = full_data[:,full_data[-1,:] != -9]
print('removal done.')

print("writing data to SRA_InputData.txt...")
np.savetxt("SRA_InputData_MAF005.txt", full_data.astype(np.int8), fmt='%i', delimiter=",")
print("all done.")         
