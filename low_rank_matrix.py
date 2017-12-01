import numpy as np
from numpy import ma as ma

import pandas as pd
import knnimpute

from masking import random_masking

datafile = 'data\winequality-red.csv'
data = pd.read_csv(datafile,delimiter=';')

print(data['fixed acidity'][10])
x = np.arange(15,1599)
data1 = data #.drop(x)
x = np.transpose(data1)
row, col = x.shape

#mask random positions in a matrix
masked_data, mask = random_masking(x)

filled_data = knnimpute.knn_impute_few_observed(masked_data,mask,5)

