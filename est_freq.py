impost numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy.ndimage

def est_freq(img, oriented_img, window_size, min_wave_length, max_wave_length):
    rows, cols = img.shape

    cos_orientation = np.mean(np.cos)