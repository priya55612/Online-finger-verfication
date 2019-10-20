import cv2
import numpy as np
from scipy import ndimage
from scipy import signal

def ridge_orientation(img, grad_sigma, orient_smooth_sigma):
    # Get size of gaussian filter for blurring. Discrete gaussian is calculated by convolving gaussian function
    # in a window equals to 6 times sigma. Value outside this window can be neglected as they are very small.
    size = np.fix(6*grad_sigma)
    if np.remainder(size,2) == 0:
        size = size+1
    gaussian = cv2.getGaussianKernel(np.int(size),grad_sigma)
    F = gaussian*gaussian.T
    F_y, F_x = np.gradient(F)
    s_x = signal.convolve2d(img, F_x, mode="same")
    s_y = signal.convolve2d(img, F_y, mode="same")

    s_xx = s_x**2
    s_yy = s_y**2
    s_xy = s_x*s_y

    # calculate theta
    # theta = 1/2 * np.arctan(scipy.convolve2d(2*s_xy, np.ones((3,3),dtype=int))/scipy.convolve2d(s_xx*s_yy, np.ones((3,3),dtype=int)))

    denominator = np.sqrt(np.power(s_xy,2)+np.power((s_xx-s_yy),2)) + np.finfo(float).eps

    #Analytic solution of principal direction
    sin_2_theta = s_xy/denominator
    cos_2_theta = (s_xx-s_yy)/denominator

    if orient_smooth_sigma:
        size = np.fix(6*orient_smooth_sigma)
        if np.remainder(size,2) == 0:
            size = size+1
        gaussian = cv2.getGaussianKernel(np.int(size), orient_smooth_sigma)
        F = gaussian*gaussian.T
        cos_2_theta = ndimage.convolve(cos_2_theta, F)
        sin_2_theta = ndimage.convolve(sin_2_theta, F)
    oriented_image = np.pi/2 + np.arctan2(sin_2_theta,cos_2_theta)/2

    return(oriented_image)
