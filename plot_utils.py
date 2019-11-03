import cv2
import numpy as np


def addDirection(decorated_img, angle, x, y, arrow_length = 15, color = (255,0,0)):
    direction_x, direction_y = (arrow_length * -1* np.cos(angle), arrow_length * np.sin(angle))  # calculate direction
    tipLength = 0.3
    start_point = (y, x)
    end_point = ( int(y + direction_y), int(x + direction_x))
    # draw arrow
    return cv2.arrowedLine(decorated_img, start_point,end_point, color, thickness=1, tipLength=tipLength)
