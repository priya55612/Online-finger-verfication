import numpy as np
from ridge import Ridge
from ridge_orientation import ridge_orientation

def find_minutiae(enhanced_image, img):
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
    sums =[]

    size_x, size_y = img.shape

    orientation_image = ridge_orientation(enhanced_image, 0.5, None)

    for i in range(1, size_x - 1):
        for j in range(1, size_y - 1):
            if img[i, j] == 1:
                sum_neighbour = (np.sum(img[i-1:i+2, j-1:j+2]))
                sums.append(sum_neighbour)
                if sum_neighbour == 2:
                    # ridge ending
                    ridge_ending.append(Ridge(i, j, orientation_image[i,j]))
                    pass
                elif sum_neighbour >= 4:
                    # ridge bifurcation
                    ridge_bifurcation.append(Ridge(i, j, orientation_image[i,j]))
                    pass
                if sum_neighbour == 3:
                    # not part of algo
                    ridge_2s.append(Ridge(i, j, orientation_image[i,j]))
    print(sums)
    print(np.max(sums), np.min(sums), np.average(sums))
    return ridge_ending, ridge_bifurcation, ridge_2s

