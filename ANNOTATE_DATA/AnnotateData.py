#!/usr/bin/python

import numpy as np
import pandas_plink

plink_files = '/media/DANE/home/jliu/SRA/MAF01/plinkmaf01'
SNP_ID_string_array_path = '/media/DANE/home/jliu/SRA/ANNOTATE_DATA/SNP_ID_string_array.csv'
SNP_ID_int_array_path = '/media/DANE/home/jliu/SRA/ANNOTATE_DATA/SNP_ID_int_array.csv'
SNP_ID_dictionary_path = '/media/DANE/home/jliu/SRA/ANNOTATE_DATA/SNP_ID_dict.txt'

# read in SNP info from plink files
snp_info  = pandas_plink.read_plink(plink_files)[0].to_numpy()

# get the string type SNP_IDs
SNP_ID_string = snp_info[:,1]

size = SNP_ID_string.shape[0]

# empty array to hold the integer type SNP IDs
print('creating integer array...')
SNP_ID_int = np.zeros(size)

int_value = int(1)

for i in range(0, size):
    SNP_ID_int[i] = int_value
    int_value = int(int_value + 1)

print('done.')

# write the two arrays to files
#print('saving string ID array...')
#np.savetxt(SNP_ID_string_array_path, SNP_ID_string, delimiter=",")
#print('done.')

print('saving integer ID array...')
np.savetxt(SNP_ID_int_array_path, SNP_ID_int.astype(np.int16), delimiter=",")
print('done.')

# creating dictionary
print('creating dictionary...')
ID_dict = {}

for ID_string, ID_int in zip(SNP_ID_string, SNP_ID_int):
    ID_dict[ID_string] = ID_int

print('done.')

# write dictionary to file
print('writing dictionary to file...')

with open(SNP_ID_dictionary_path, 'a') as f:
    for key, value in ID_dict.items():
        f.write('%s:%i\n' % (key, value))

print('all done.')



