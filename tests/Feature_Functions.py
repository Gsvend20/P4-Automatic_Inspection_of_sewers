import cv2 as cv
import pathlib
import numpy as np

# Select picture path from .../P4-Automatic_Inspection_of_sewers/ directory
picPath = '/materials/GR-type_0_2.jpg'


def find_circularity(area, perimeter):
    circ = 4*np.pi*area/pow(perimeter, 2)
    return circ


def find_compactness(area, width, height):
    comp = area/(width*height)
    return comp


def find_elongation(cnt):
    (x, y), (width, height), angle = cv.minAreaRect(cnt)
    elongation = min(width, height) / max(width, height)
    return elongation

def find_ferets(cnt):
    (x, y), (width, height), angle = cv.minAreaRect(cnt)
    if width >= height:
        return width
    else:
        return height

def find_intensity(cnt, gray_img):
    mask = np.zeros(gray_img.shape, np.uint8)
    cv.drawContours(mask, [cnt], 0, 255, -1)
    mean_val = cv.mean(gray_img, mask=mask)
    return mean_val

# Simple version of thinness
def find_thinness(cnt):
    # compute the area of the contour along with the bounding box
    # to compute the aspect ratio
    area = cv.contourArea(cnt)
    circum = cv.arcLength(cnt, True)

    thinness = (4 * np.pi * area) / (circum**2)
    return thinness

cnt = 'This is your contour'

# Aprox of the shape, none precise method
contours_poly = cv.approxPolyDP(cnt, 3, True)

# Approx precise method
# The lower the epsilon value the more precise the outcome
epsilon = 0.001 * cv.arcLength(cnt, True)
approximations = cv.approxPolyDP(cnt, epsilon, True)


# Bounding box
boundRect = cv.boundingRect(contours_poly)

# Enclosing circle
centers, radius = cv.minEnclosingCircle(contours_poly)



