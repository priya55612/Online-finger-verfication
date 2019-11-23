from normalization import ridge_segementation
from ridge_orientation import ridge_orientation
from ridge_frequency import ridge_frequency
from ridge_filter import ridge_filter
from skimage import morphology
import numpy as np
import cv2


def img_enhancement(img):
    clahe = cv2.createCLAHE(clipLimit=2.0,
                            tileGridSize=(8, 8))  # contrast enhancement using adaptive histogram equalization
    img = clahe.apply(img)
    # cv2.imshow("clahe img", img)

    window_size = 16
    threshold = 0.1
    norm_img, mask = ridge_segementation(img, window_size, threshold)
    # cv2.imshow("normal image", norm_img)

    grad_sigma = 1
    window_size = 7
    orient_smooth_sigma = 7
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


def img_skeleton(img):
    # Remove breaks in ridges by applying closing operation. This is a novelty of we can achieve this
    closed_image = morphology.closing(img, np.ones((3, 3)))
    cv2.imshow("closed Image", closed_image.astype('int') * 255.0)

    skeleton_image = morphology.skeletonize(img)
    return skeleton_image


def remove_noise(img):
    input_img = np.array(img.astype('int')*1.0)
    output_img = np.array(img.astype('int')*1.0)

    W, H = input_img.shape[:2]
    filtersizes = [4, 6, 8, 12, 20]
    filtersizes = [4, 20]
    for filtersize in filtersizes:
        for i in range(W - filtersize):
            for j in range(H - filtersize):
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
    img = img_enhancement(img)
    cv2.imshow("enhanced image", img.astype('int') * 255.0)
    # Thinning
    skeleton = img_skeleton(img)
    cv2.imshow("skeleton image output", skeleton.astype('int') * 255.0)

    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = remove_noise(skeleton)
    cv2.imshow("skeleton after removing noise", skeleton.astype('int') * 255.0)
    return skeleton
