import cv2
import numpy as np

def sift(img, ridge_endings, ridge_bifurcations, ridge_2s):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img,None)
    img=cv2.drawKeypoints(img,kp,img)
    cv2.imshow("sift ", img)
