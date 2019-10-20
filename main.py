"""This is the start file for project. Code execution flow start from here."""
import cv2
import image_enhancement
import input_image
import minutiae_util
w = 3

def main():
    input_im = input_image.input_im()
    cv2.imshow("original image", input_im)

    enhanced_image = image_enhancement.img_enhancement(input_im)
    skeleton_image = image_enhancement.img_skeleton(enhanced_image)

    cv2.imshow("enhanced skeleton image", skeleton_image.astype('int')*255.0)

    ridge_endings, ridge_bifurcations, ridge_2s = minutiae_util.find_minutiae(skeleton_image)
    print(ridge_endings)
    print(ridge_bifurcations)
    print(ridge_2s)
    print(len(ridge_endings))
    print(len(ridge_bifurcations))
    print(len(ridge_2s))

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
