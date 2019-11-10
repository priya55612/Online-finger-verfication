"""This is the start file for project. Code execution flow start from here."""
import cv2
import image_enhancement
import input_image
import minutiae_util

def main():
    input_im = input_image.input_im()
    cv2.imshow("original image", input_im)

    enhanced_image = image_enhancement.img_enhancement(input_im)
    skeleton_image = image_enhancement.img_skeleton(enhanced_image)

    cv2.imshow("enhanced_image", enhanced_image.astype("int") * 255.0)
    cv2.imshow("enhanced skeleton image", skeleton_image.astype('int') * 255.0)

    # remove secluded points
    imgae_wo_noise = image_enhancement.remove_noise(skeleton_image)
    cv2.imshow("noise free image", imgae_wo_noise)

    ridge_endings, ridge_bifurcations, ridge_2s = minutiae_util.find_minutiae(imgae_wo_noise)
    minutiae_util.plot_minutiae(imgae_wo_noise, ridge_endings, ridge_bifurcations)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
