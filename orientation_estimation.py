import cv2
import numpy as np
import scipy.signal as scipy
def getLocalGradient(im, w, filter = 'sobel'):
    '''
    :param im: Image
    :param w: size of local window
    :param filter: filter type
    :return: localOrientation
    '''
    height, width = im.shape
    # Compute Gradient
    gx = cv2.Sobel(im, -1, 1, 0, w)
    gy = cv2.Sobel(im, -1, 0, 1, w)
    # cv2.imshow("x", gx)
    # cv2.imshow("y", gy)
    # return gx, gy
    # Compute local orientation
    return getLocalOrientation(gx,gy, w, height, width)

def getLocalOrientation(gx, gy, w, height, width):
    window = int(w/2)
    gx_mult_gy = 2 * gx*gy
    gx2_subt_gy2 = gx*gx - gy*gy
    x, y = np.meshgrid(np.arange(1,height-1), np.arange(1,width-1))
    return 1/2 * np.arctan(scipy.convolve2d(gx_mult_gy, np.ones((3,3),dtype=int))/scipy.convolve2d(gx2_subt_gy2, np.ones((3,3),dtype=int)))


# print(x,y)
