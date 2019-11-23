from normalization import ridge_segementation
from ridge_orientation import ridge_orientation
from ridge_frequency import ridge_frequency
from ridge_filter import ridge_filter
from skimage import morphology
import numpy as np
import cv2


# Algorithm reference paper
# '''Fingerprint Image Enhancement: Algorithm and Performance Evaluation'''
# https://pdfs.semanticscholar.org/bd6d/e6c7fba04a67d30a4bd1261665e6f4745ea8.pdf
def enhance_image(img):
    window_size = 16
    threshold = 0.1
    # Gnerate fingerpringt mask from given image
    norm_img, mask = ridge_segementation(img, window_size, threshold)
    cv2.imshow("normal image", norm_img)

    grad_sigma = 1
    orient_smooth_sigma = 7
    # Find orientation of each pixel by using gradient of its surrounding
    oriented_img = ridge_orientation(img, grad_sigma, orient_smooth_sigma)
    # cv2.imshow("orientation image", oriented_img)

    block_size = 38
    window_size = 5
    min_wave_length = 5
    max_wave_length = 15
    freq, mean_freq = ridge_frequency(norm_img, mask, oriented_img, block_size, window_size, min_wave_length,
                                      max_wave_length)
    # cv2.imshow("freauency image", freq)

    freq = mean_freq * mask
    k_x = 0.65
    k_y = 0.65
    new_img = ridge_filter(norm_img, oriented_img, freq, k_x, k_y)
    # cv2.imshow("filter image", new_img)

    return new_img < -3


def skeletonize(img):
    # Remove breaks in ridges by applying closing operation.
    closed_image = morphology.closing(img, np.ones((3, 3)))
    cv2.imshow("closed Image", closed_image.astype('int') * 255.0)

    skeleton_image = morphology.skeletonize(closed_image)
    return skeleton_image


# This function checks if a window has no foreground pixels on 3 or more boundaries. If so, mark the complete window background.
# This removes any secluded point in image.
# Repeat this with multiple windows sizes.
def remove_noise(img):
    input_img = np.array(img.astype('int') * 1.0)
    output_img = np.array(img.astype('int') * 1.0)

    W, H = input_img.shape[:2]
    filtersizes = [4, 6, 8, 12, 20]
    # filtersizes = [4, 20]
    for filtersize in filtersizes:
        for i in range(0, W - filtersize, filtersize):
            for j in range(0, H - filtersize, filtersize):
                filter0 = output_img[i:i + filtersize, j:j + filtersize]

                flag = 0
                if sum(filter0[:, 0]) == 0:
                    flag += 1
                if sum(filter0[:, filtersize - 1]) == 0:
                    flag += 1
                if sum(filter0[0, :]) == 0:
                    flag += 1
                if sum(filter0[filtersize - 1, :]) == 0:
                    flag += 1
                if flag > 3:
                    output_img[i:i + filtersize, j:j + filtersize] = np.zeros((filtersize, filtersize))

    return output_img


def perform_enhancement_and_morphology(img):
    enhanced_img = enhance_image(img)
    cv2.imshow("enhanced image", enhanced_img.astype('int') * 255.0)
    # Thinning
    skeleton_img = skeletonize(enhanced_img)
    cv2.imshow("skeleton image output", skeleton_img.astype('int') * 255.0)
    # Remove the breaks and secluded elements in fingerprint.
    skeleton_image_wo_noise = remove_noise(skeleton_img)
    cv2.imshow("skeleton after removing noise", skeleton_image_wo_noise.astype('int') * 255.0)
    return skeleton_image_wo_noise
