from normalization import ridge_segementation
from ridge_orientation import ridge_orientation
from ridge_frequency import ridge_frequency
from ridge_filter import ridge_filter
from skimage import morphology
import cv2


def img_enhancement(img):
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
    freq, mean_freq = ridge_frequency(norm_img, mask, oriented_img, block_size, window_size, min_wave_length, max_wave_length)
    # cv2.imshow("freauency image", freq)

    freq = mean_freq*mask
    k_x = 0.65
    k_y = 0.65
    new_img = ridge_filter(norm_img, oriented_img, freq, k_x, k_y)
    # cv2.imshow("filter image", new_img)

    return new_img < -3

def img_skeleton(img):
    # TODO: Remove breaks in ridges by applying closing operation. This is a novelty of we can achieve this
    # closed_image = morphology.closing(enhanced_image, np.ones((3,3)))
    # cv2.imshow("closed Image", closed_image.astype('int')*255.0)

    # cv2.imshow("enanced image", enhanced_image.astype('int')*255.0)
    skeleton_image = morphology.skeletonize(img)
    return skeleton_image


