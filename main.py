"""This is the start file for project. Code execution flow start from here."""
import cv2

import descriptors
import image_enhancement
import input_image


# This is the main entry point for Online finger print verification.
# Briefly, algorithm involves following steps
# 1. Reading 2 finger prints
# 2. Enhancement, restoration and morphological processing
# 3. Get minutiae from the binary, skeletonized image obtained in previous step.
# 4. Get descriptors for minutiae points
# 5. Matching using Brute force matcher
def main():
    # Read two images. PLEASE provide complete path containing test_input folder
    img_1 = input_image.input_im('test_input/1/01_1.bmp')
    img_2 = input_image.input_im('test_input/1/01_10.bmp')

    # Enhancing the image, followed by morphological operations.
    img_skeleton_1 = image_enhancement.perform_enhancement_and_morphology(img_1)
    img_skeleton_2 = image_enhancement.perform_enhancement_and_morphology(img_2)

    # This step finds ridge bifurcations and endings. Descriptors are obtained for these minutiae points
    kp1, des1 = descriptors.get_descriptors(img_skeleton_1)
    kp2, des2 = descriptors.get_descriptors(img_skeleton_2)

    # Displaying key points(ridge bifurcations and endings) on image
    img4 = cv2.drawKeypoints(img_skeleton_1.astype('uint8') * 255, kp1, outImage=None)
    img5 = cv2.drawKeypoints(img_skeleton_2.astype('uint8') * 255, kp2, outImage=None)
    cv2.imshow("key points for img_1", img4)
    cv2.imshow("key points for img_2", img5)

    # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)
    # Plot keypoints

    # Plot matches
    img3 = cv2.drawMatches(img_skeleton_1.astype('uint8') * 255, kp1, img_skeleton_2.astype('uint8') * 255, kp2,
                           matches, flags=2, outImg=None)
    cv2.imshow("matched image ", img3)
    cv2.waitKey(0)

    # Calculate score and display matching result.
    score = 0
    for match in matches:
        score += match.distance
    score_threshold = 33
    print(score_threshold)
    if score / len(matches) < score_threshold:
        print("Fingerprint matches.")
    else:
        print("Fingerprint does not match.")


if __name__ == '__main__':
    main()
