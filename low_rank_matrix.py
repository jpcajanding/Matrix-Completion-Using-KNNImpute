import numpy as np
from numpy import ma as ma

import pandas as pd
import knnimpute

from masking import random_masking, time_masking

datafile = 'data\winequality-red.csv'
data = pd.read_csv(datafile,delimiter=';')

# print(data['fixed acidity'][10])
print(data.shape)
# data1 = data #.drop(x)
masked_data, mask = time_masking(data,mask_time=0.01)
masked_data = np.transpose(masked_data)
mask = np.transpose(mask)

#mask random positions in a matrix
# masked_data, mask = random_masking(x)

# filled_data = knnimpute.knn_impute_few_observed(masked_data,mask,5)
filled_data = knnimpute.knn_impute_with_argpartition(masked_data,mask,5)

print(np.linalg.norm((data-(np.transpose(filled_data)))))