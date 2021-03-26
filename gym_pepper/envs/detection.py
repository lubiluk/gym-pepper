import cv2
import numpy as np

lower_range = np.array([110, 50, 50])
upper_range = np.array([130, 255, 255])


def is_object_in_sight(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)

    cv2.imshow("camera", img)
    cv2.imshow('mask', mask)

    return mask.max() != 0
