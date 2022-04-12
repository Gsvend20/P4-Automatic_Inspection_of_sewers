from __future__ import print_function
import cv2 as cv
import pathlib
import numpy as np

# --- To stop program press q ---

# Select picture path from .../P4-Automatic_Inspection_of_sewers/ directory
picPath = '/materials/GR-type_0_2.jpg'


def find_circularity(area, perimeter):
    circ = 4*np.pi*area/pow(perimeter, 2)
    return circ


def find_compactness(area, width, height):
    comp = area/(width*height)
    return comp


def find_elongation(cnt):
    (x, y), (width, height), angle = minAreaRect(cnt)
    elongation = min(width, height) / max(width, height)
    return elongation

def find_ferets(cnt):
    (x, y), (width, height), angle = minAreaRect(cnt)
    if width >= height:
        return width
    else:
        return height

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



