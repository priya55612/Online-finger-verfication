import numpy as np
from  est_freq import est_freq


def ridge_frequency(img, mask, orientation, block_size, window_size, min_wave_length, max_wave_length):
    rows, cols = img.shape
    frequency = np.zeros((rows, cols))

    for i in range(0, rows-block_size, block_size):
        for j in range(0, cols-block_size, block_size):
            block_image = img[i:i+block_size][:,j:j+block_size]
            block_orientation = orientation[i:i+block_size][:,j:j+block_size]

            frequency[i:i+block_size][:,j:j+block_size] = est_freq(block_image, block_orientation, window_size, min_wave_length, max_wave_length)
    frequency = frequency*mask

    frequency_1d = np.reshape(frequency,(1, rows*cols))

    index = np.where(frequency_1d>0)

    index = np.array(index)

    index = index[1,:]

    non_zero_ele_in_frequency = frequency_1d[0][index]

    mean_frequency = np.mean(non_zero_ele_in_frequency)

    median_frequency = np.median(non_zero_ele_in_frequency)

    return(frequency, mean_frequency)