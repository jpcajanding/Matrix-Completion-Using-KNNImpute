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

def time_masking(data, masked_node = 2, mask_time = 0.1, resolution = 0.004, start_percentage = 0.10):
    row, col = data.shape

    #the matrix is an 9xn matrix where n is the total number of data.
    #need to mask a block of 3xm where m is dependent on the resolution
    start_index = int(row * start_percentage)
    num_mask = int(mask_time/resolution)
    if start_index + num_mask < row:
        mask = np.zeros((row,col))

        #columns 0-2 for node 1, 3-5 for node 2 and cols 6-8 for node 3
        mask_cols = np.array([3*(masked_node-1),3*(masked_node-1)+1,3*(masked_node-1)+2])
        masking = np.ones((num_mask,3))
        print(masking.shape)
        mask[start_index:(start_index+num_mask),(3*(masked_node-1)):(3*(masked_node-1)+3)] = masking
        masked_data = ma.masked_array(data, mask)
        mask = masked_data.mask

        masked_data = masked_data.filled(np.NaN)

    return masked_data, mask