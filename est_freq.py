import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy.ndimage

def est_freq(img, oriented_img, window_size, min_wave_length, max_wave_length):
    rows, cols = img.shape

    cos_orientation = np.mean(np.cos(2*oriented_img))
    sin_orientation = np.mean(np.sin(2*oriented_img))

    orientation = math.atan2(sin_orientation, cos_orientation)/2

    rotated_img = scipy.ndimage.rotate(img, orientation/np.pi*180 +90, axes=(1,0), reshape=False, order=3, mode='nearest')


    crop_size = int(np.fix(rows/np.sqrt(2)))
    off_set = int(np.fix((rows-crop_size)/2))

    rotated_img = rotated_img[off_set:off_set+crop_size][:,off_set:off_set+crop_size]

    projection = np.sum(rotated_img, axis=0)

    dilation = scipy.ndimage.grey_dilation(projection, window_size, structure=np.ones(window_size))

    temp = np.abs(dilation - projection)

    threshold_peak = 2

    max_pts = (temp<threshold_peak) & (projection>np.mean(projection))

    max_index = np.where(max_pts)

    rows_max_index, cols_max_index = np.shape(max_index)

    if(cols_max_index<2):
        freq_img = cols_max_index
    else:
        num_of_peaks = cols_max_index
        wave_length = (max_index[0][cols_max_index-1]-max_index[0][0])/(num_of_peaks-1)

        if wave_length>=min_wave_length and wave_length<=max_wave_length:
            freq_img = 1/np.double(wave_length)*np.ones(img.shape)
        else:
            freq_img = np.zeros(img.shape)
    return(freq_img)