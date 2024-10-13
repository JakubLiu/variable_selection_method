import numpy as np

full_dataset = np.loadtxt('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/NN_input_data_FULL.txt')

# splitting the dataset into the learn and validation part
np.random.shuffle(full_dataset)
cutoff = int(full_dataset.shape[1]*0.8)
data_to_learn = full_dataset[:cutoff, :]
validation_set = full_dataset[cutoff:full_dataset.shape[1], :]

# downsampling the more prevalent phenotype in the validation set (the learn set is left imbalanced)
phenotype1 = validation_set[validation_set[:,-1] == 1.0,]
phenotype0 = validation_set[validation_set[:,-1] == 0.0,]
phenotype0_small = phenotype0[:phenotype1.shape[0],:]
validation_set_balanced = np.concatenate((phenotype0_small, phenotype1), axis = 0)

# saving
np.savetxt('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/NN_input_data_LEARN.txt', data_to_learn)
np.savetxt('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/NN_input_data_VALIDATION_IMBALANCED.txt', validation_set)
np.savetxt('/media/DANE/home/jliu/SRA/PREDICTION/NEURAL_NETWORK/NN_input_data_VALIDATION_BALANCED.txt', validation_set_balanced)

print('done.')
