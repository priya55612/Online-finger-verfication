"""This is the start file for project. Code execution flow start from here."""
import cv2
import image_enhancement
import input_image
import minutiae_util
import numpy as np


def addDirection(decorated_img, angle, x, y):
    arrow_length = 15
    direction_x, direction_y = (arrow_length * -1* np.cos(angle), arrow_length * np.sin(angle))  # calculate direction
    tipLength = 0.3
    start_point = (y, x)
    end_point = ( int(y + direction_y), int(x + direction_x))
    # draw arrow
    return cv2.arrowedLine(decorated_img, start_point,end_point, (255, 0, 0), thickness=1, tipLength=tipLength)


def main():
    input_im = input_image.input_im()
    cv2.imshow("original image", input_im)

    enhanced_image = image_enhancement.img_enhancement(input_im)
    skeleton_image = image_enhancement.img_skeleton(enhanced_image)

    cv2.imshow("enhanced_image", enhanced_image.astype("int")*255.0)
    cv2.imshow("enhanced skeleton image", skeleton_image.astype('int')*255.0)

    ridge_endings, ridge_bifurcations, ridge_2s = minutiae_util.find_minutiae(enhanced_image, skeleton_image)
    angles = []
    decorated_img = cv2.cvtColor(skeleton_image.astype('uint8')*255, cv2.COLOR_GRAY2RGB)
    for i in range(len(ridge_bifurcations)):
        x, y = ridge_bifurcations[i].x, ridge_bifurcations[i].y
        decorated_img[x][y] = (0,255,0)
        # decorated_img = addDirection(decorated_img, ridge_bifurcations[i].orientation, x, y)
        # angles.append(ridge_bifurcations[i].orientation)
    for i in range(len(ridge_endings)):
        x, y = ridge_endings[i].x, ridge_endings[i].y
        decorated_img[x][y] = (0,0,255)
        decorated_img = addDirection(decorated_img, ridge_endings[i].orientation, x, y)
        # angles.append(ridge_endings[i].orientation)
    cv2.imshow("decorated image pts", decorated_img)
    print("angles", np.unique(angles))
    print("angles len", len(np.unique(angles)))
    print(ridge_endings)
    print(ridge_bifurcations)
    print(ridge_2s)
    print(len(ridge_endings))
    print(len(ridge_bifurcations))
    print(len(ridge_2s))

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
