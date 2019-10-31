"""This is the start file for project. Code execution flow start from here."""
import cv2
import image_enhancement
import input_image
import minutiae_util
import numpy as np
w = 3

def main():
    input_im = input_image.input_im()
    cv2.imshow("original image", input_im)

    enhanced_image = image_enhancement.img_enhancement(input_im)
    skeleton_image = image_enhancement.img_skeleton(enhanced_image)

    cv2.imshow("enhanced_image", enhanced_image.astype("int")*255.0)
    cv2.imshow("enhanced skeleton image", skeleton_image.astype('int')*255.0)

    ridge_endings, ridge_bifurcations, ridge_2s = minutiae_util.find_minutiae(skeleton_image)

    ending_img = np.zeros(input_im.shape)
    # bifurcation_img = np.zeros(input_im.shape, dtype="uint8")
    ending_img = cv2.cvtColor(np.zeros(input_im.shape, dtype="uint8"), cv2.COLOR_GRAY2RGB)
    bifurcation_img = cv2.cvtColor(skeleton_image.astype('uint8')*255, cv2.COLOR_GRAY2RGB)
    for i in range(len(ridge_bifurcations)):
        x, y = ridge_bifurcations[i].x, ridge_bifurcations[i].y
        bifurcation_img[x][y] = (0,255,0)
    for i in range(len(ridge_endings)):
        x, y = ridge_endings[i].x, ridge_endings[i].y
        ending_img[x][y] = (0,0,255)
    cv2.imshow("bifurcation pts", bifurcation_img)
    cv2.imshow("ending pts", ending_img)



    print(ridge_endings)
    print(ridge_bifurcations)
    print(ridge_2s)
    print(len(ridge_endings))
    print(len(ridge_bifurcations))
    print(len(ridge_2s))

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
