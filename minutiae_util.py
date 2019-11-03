import numpy as np
from ridge import Ridge
from ridge_orientation import ridge_orientation
import cv2
import plot_utils
from matplotlib import pyplot as plt


def plot_minutiae(skeleton_image, ridge_endings, ridge_bifurcations):
    # angles = []
    decorated_img = cv2.cvtColor(skeleton_image.astype('uint8') * 255, cv2.COLOR_GRAY2RGB)
    for i in range(len(ridge_bifurcations)):
        x, y = ridge_bifurcations[i].x, ridge_bifurcations[i].y
        decorated_img = plot_utils.addDirection(decorated_img, ridge_bifurcations[i].orientation, x, y, color=(0,255,255))
        # angles.append(ridge_bifurcations[i].orientation)
    for i in range(len(ridge_endings)):
        x, y = ridge_endings[i].x, ridge_endings[i].y
        decorated_img = plot_utils.addDirection(decorated_img, ridge_endings[i].orientation, x, y)
        # angles.append(ridge_endings[i].orientation)
    cv2.imshow("decorated image pts", decorated_img)
    # print("angles", np.unique(angles))
    # print("angles len", len(np.unique(angles)))
    # print(ridge_endings)
    # print(ridge_bifurcations)
    # print(len(ridge_endings))
    # print(len(ridge_bifurcations))


def find_minutiae(img):
    """
    This function find minutia in skeleton image. minutia is detected as follows.
    1. If a pixel has exactly one foreground neighbor, its a ridge ending
    2. If a pixel has exactly two foreground neighbor, its a pixel on ridge
    3. If a pixel has more than two foreground neighbor, its an ridge bifurcation pixel
    :type img: skeleton image from which minutiae is need to be found
    """

    ridge_ending = []
    ridge_bifurcation = []
    ridge_2s = []
    sums = []

    size_x, size_y = img.shape

    orientation_image = ridge_orientation(img, 1, 5) - np.pi / 2
    # cv2.imshow("orientation image", (orientation_image*255/np.pi).astype('uint8'))

    for i in range(1, size_x - 1):
        for j in range(1, size_y - 1):
            if img[i, j] == 1:
                sum_neighbour = (np.sum(img[i - 1:i + 2, j - 1:j + 2]))
                sums.append(sum_neighbour)
                if sum_neighbour == 2:
                    # ridge ending
                    ridge_ending.append(Ridge(i, j, orientation_image[i, j]))
                    pass
                elif sum_neighbour >= 4:
                    # ridge bifurcation
                    ridge_bifurcation.append(Ridge(i, j, orientation_image[i, j]))
                    pass
                if sum_neighbour == 3:
                    # not part of algo
                    ridge_2s.append(Ridge(i, j, orientation_image[i, j]))
    print(sums)
    print(np.max(sums), np.min(sums), np.average(sums))
    return ridge_ending, ridge_bifurcation, ridge_2s
