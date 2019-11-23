import cv2
import numpy as np
import image_enhancement
import minutiae_util


def get_descriptors(skeleton):
    ridge_endings, ridge_bifurcations, ridge_2s = minutiae_util.find_minutiae(skeleton)
    # Extract keypoints
    keypoints = []
    for i in range(len(ridge_bifurcations)):
        x, y = ridge_bifurcations[i].x, ridge_bifurcations[i].y
        keypoints.append(cv2.KeyPoint(y, x, 1))
    # for i in range(len(ridge_endings)):
    #     x, y = ridge_endings[i].x, ridge_endings[i].y
    #     keypoints.append(cv2.KeyPoint(x, y, 1))
    # Define descriptor
    orb = cv2.ORB_create()
    # Compute descriptors
    _, des = orb.compute(skeleton.astype('uint8')*255, keypoints)
    return (keypoints[:], des)


    # Harris corners
    # harris_corners = cv2.cornerHarris(img, 3, 3, 0.04)
    # harris_normalized = cv2.normalize(harris_corners, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
    # threshold_harris = 125

    # for x in range(0, harris_normalized.shape[0]):
    #     for y in range(0, harris_normalized.shape[1]):
    #         if harris_normalized[x][y] > threshold_harris:
    #             keypoints.append(cv2.KeyPoint(y, x, 1))
