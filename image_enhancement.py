from normalization import ridge_segementation
from ridge_orientation import ridge_orientation
from ridge_frequency import ridge_frequency
from ridge_filter import ridge_filter


def img_enhancement(img):
    window_size = 16
    threshold = 0.1
    norm_img, mask = ridge_segementation(img, window_size, threshold)

    grad_sigma = 1
    window_size = 7
    orient_smooth_sigma = 7
    oriented_img = ridge_orientation(img, grad_sigma, window_size, orient_smooth_sigma)

    block_size = 38
    window_size = 5
    min_wave_length = 5
    max_wave_length = 15
    freq, mean_freq = ridge_frequency(norm_img, mask, oriented_img, block_size, window_size, min_wave_length, max_wave_length)

    freq = mean_freq*mask
    k_x = 0.65
    k_y = 0.65
    new_img = ridge_filter(norm_img, oriented_img, freq, k_x, k_y)

    return(new_img < -3)

