"""This is the start file for project. Code execution flow start from here."""
import cv2
import image_enhancement
import input_image
import minutiae_util
import descriptors
import matplotlib.pyplot as plt


def main():
    # input_im = input_image.input_im()
    # cv2.imshow("original image", input_im)
    #
    # enhanced_image = image_enhancement.img_enhancement(input_im)
    # skeleton_image = image_enhancement.img_skeleton(enhanced_image)
    #
    # cv2.imshow("enhanced_image", enhanced_image.astype("int") * 255.0)
    # cv2.imshow("enhanced skeleton image", skeleton_image.astype('int') * 255.0)
    #
    # # remove secluded points
    # imgae_wo_noise = image_enhancement.remove_noise(skeleton_image)
    # cv2.imshow("noise free image", imgae_wo_noise)
    #
    # ridge_endings, ridge_bifurcations, ridge_2s = minutiae_util.find_minutiae(imgae_wo_noise)
    # minutiae_util.plot_minutiae(imgae_wo_noise, ridge_endings, ridge_bifurcations)
    # cv2.waitKey(0)
    img_1 = cv2.cvtColor(cv2.imread('C:/Users/Priyanka/Desktop/DIP_project/Online-finger-verfication/88.png'), cv2.COLOR_RGB2GRAY)
    img_2 = cv2.cvtColor(cv2.imread('C:/Users/Priyanka/Desktop/DIP_project/Online-finger-verfication/1_3.png'), cv2.COLOR_RGB2GRAY)

    kp1, des1 = descriptors.get_descriptors(img_1)
    kp2, des2 = descriptors.get_descriptors(img_2)

    # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)
    # Plot keypoints
    img4 = cv2.drawKeypoints(img_1, kp1, outImage=None)
    img5 = cv2.drawKeypoints(img_2, kp2, outImage=None)
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img4)
    axarr[1].imshow(img5)
    plt.show()
    # Plot matches
    img3 = cv2.drawMatches(img_1, kp1, img_2, kp2, matches, flags=2, outImg=None)
    plt.imshow(img3)
    plt.show()

    # Calculate score
    score = 0
    for match in matches:
        score += match.distance
    score_threshold = 33
    if score / len(matches) < score_threshold:
        print("Fingerprint matches.")
    else:
        print("Fingerprint does not match.")







if __name__ == '__main__':
    main()
