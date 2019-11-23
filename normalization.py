import cv2
import numpy as np
import matplotlib.pyplot as plt


def normalize(img):
    norm_img = (img-np.mean(img))/np.std(img)
    return norm_img

# Below function returns the masked image with the given threshold and normalized image
def ridge_segementation(image, window_size, threshold):
    rows, cols = image.shape
    image = normalize(image)

    padded_rows = np.int(window_size * np.ceil((np.float(rows))/(np.float(window_size))))
    padded_cols = np.int(window_size * np.ceil((np.float(cols)) / (np.float(window_size))))

    padded_img = np.zeros((padded_rows, padded_cols))

    std_img = np.zeros((padded_rows, padded_cols))

    padded_img[0:rows][:, 0:cols] = image

    for i in range(0, padded_rows, window_size):
        for j in range(0, padded_cols, window_size):
            window = padded_img[i:i+window_size][:,j:j+window_size]
            std_img[i:i+window_size][:,j:j+window_size] = np.std(window)*np.ones(window.shape)

    std_img = std_img[0:rows][:,0:cols]


    mask_img = std_img > threshold

    mean_val = np.mean(image[mask_img])
    std_val = np.std(image[mask_img])

    norm_image = (image - mean_val) / (std_val)

    return (norm_image, mask_img)






