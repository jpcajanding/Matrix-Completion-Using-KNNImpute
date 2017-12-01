import numpy as np
from numpy import ma as ma

def random_masking(data, num_mask=10):
    row, col = data.shape
    mask = np.array([1] * num_mask + [0] * ((row * col) - num_mask))
    np.random.shuffle(mask)
    mask = np.reshape(mask, data.shape)

    masked_data = ma.masked_array(data, mask)
    mask = masked_data.mask

    masked_data = masked_data.filled(np.NaN)
    return masked_data, mask

def time_masking(data, masked_row = 1, time = 0.1, resolution = 0.004):
    row, col = data.shape