"""This is the start file for project. Code execution flow start from here."""
import cv2
import numpy as np
import orientation_estimation as oe
import normalization as segmentation
import ridge_orientation as rio

w = 3

def main():
    input_im = cv2.cvtColor(cv2.imread('C:/Users/Priyanka/Desktop/DIP_project/Online-finger-verfication/fingerprint.jpg'), cv2.COLOR_RGB2GRAY)
    # Estimating Local orientation at each place
    # localOrientation = oe.getLocalGradient(input_im.copy(), w)
    # cv2.imshow("localOrientation", localOrientation)
    # cv2.waitKey(0)
    segmentedImage, mask = segmentation.ridge_segementation(input_im, 16,0.2)
    cv2.imshow("segmentedImage", segmentedImage)
    cv2.imshow("inputImage", input_im)
    # cv2.waitKey(0)
    print(mask.shape)
    cv2.imshow("mask", (mask.astype(int)*255.0))
    # cv2.waitKey(0)

    ridge_orientation_img = rio.ridge_orientation(segmentedImage, 1, 7, None)
    print(np.max(ridge_orientation_img))
    print(np.min(ridge_orientation_img))
    # cv2.imshow("ridge_orientation_img", ridge_orientation_img)
    cv2.waitKey(0)
    #print(input_im.shape)


if __name__ == '__main__':
    main()
