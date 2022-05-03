from os import listdir
import cv2
import numpy as np
from imgproc_func import resize_image


class FeatureSpace:
    def __init__(self):
        self.matrix = []
        self.centerX = []
        self.centerY = []
        self.convex_ratio_perimeter = []
        self.hierachy_Bool = []
        self.compactness = []
        self.elongation = []

    def createFeauteres(self, image):
        img = cv2.imread(image, 0)
        if img is not None and np.mean(img) > 0:
            contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            hierarchy = hierarchy[0]
            n = 0
            for k in range(len(contours)):
                area = cv2.contourArea(contours[k])
                if area > n:
                    n = area
                    cnt = contours[k]
                    hrc = np.array(hierarchy[k][2] != -1)
            M = cv2.moments(cnt)
            self.centerX.append(int(M['m10'] / M['m00']))
            self.centerY.append(M['m01'] / M['m00'])
            perimeter = cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            hullperimeter = cv2.arcLength(hull, True)
            self.convex_ratio_perimeter.append(int(perimeter/hullperimeter))
            self.compactness.append(int(n/img.shape[0]*img.shape[1]))
            if hrc:
                self.hierachy_Bool.append(1)
            else:
                self.hierachy_Bool.append(0)


sand = FeatureSpace()
types = [15, 55, 120, 150]
folder = "Sandtraindata"
for i in types:
        mask = listdir(f"{folder}/{i}/bgr/rgbMasks")
        for j in mask:
            checkpng = j.find("png")
            if checkpng >= 1:
                sand.createFeauteres(f"{folder}/{i}/bgr/rgbMasks/{j}")
