"""This is the start file for project. Code execution flow start from here."""
import cv2
import numpy as np
import orientation_estimation as oe

w = 3

def main():
    input_im = cv2.cvtColor(cv2.imread('finger_print.jpg'), cv2.COLOR_RGB2GRAY)
    # Estimating Local orientation at each place
    localOrientation = oe.getLocalGradient(input_im, w)
    cv2.imshow("localOrientation", localOrientation)
    cv2.waitKey(0)
    print(input_im.shape)


if __name__ == '__main__':
    main()
