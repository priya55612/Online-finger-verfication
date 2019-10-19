import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal

def ridge_orientation(img, grad_sigma, window_sigma, orient_smooth_sigma):
    size = np.fix(6*grad_sigma)
    if np.remainder(size,2) == 0:
        size = size+1
    gaussian = cv2.getGaussianKernel(np.int(size),grad_sigma)
    F = gaussian*gaussian.T
    F_y, F_x = np.gradient(F)
    cv2.imshow("cgrafient ", F_x)
    cv2.imshow("cgrafient2 ", F_y)
    G_x = signal.convolve2d(img, F_x, mode="same")
    G_y = signal.convolve2d(img, F_y, mode="same")

    G_xx = np.power(G_x,2)
    G_yy = np.power(G_y, 2)
    G_xy = G_x*G_y

    size = np.fix(6*window_sigma)

    gaussian = cv2.getGaussianKernel(np.int(size), window_sigma)
    F = gaussian*gaussian.T

    G_xx = ndimage.convolve(G_xx,F)
    G_yy = ndimage.convolve(G_yy,F)
    G_xy = 2*ndimage.convolve(G_xy,F)

    denominator = np.sqrt(np.power(G_xy,2)+np.power((G_xx-G_yy),2)) + np.finfo(float).eps

    sin_2_theta = G_xy/denominator
    cos_2_theta = (G_xx-G_yy)/denominator

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