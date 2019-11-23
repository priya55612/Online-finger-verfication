import cv2

def input_im(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2GRAY);
