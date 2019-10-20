import cv2

def input_im():
    return cv2.cvtColor(cv2.imread('finger_print.jpg'), cv2.COLOR_RGB2GRAY);
