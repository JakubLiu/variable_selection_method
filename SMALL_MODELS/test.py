#!/usr/bin/bash

import numpy as np

mat = np.zeros((100, 100))

np.savetxt('/media/DANE/home/jliu/SRA/SMALL_MODELS/test.txt', mat)

print('done.')
