import cv2 as cv
from pathlib import Path
import numpy as np

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(image_name, int(width), int(height))
    cv.imshow(image_name, image)


picPath = Path.cwd().parent.as_posix()+'/materials/GR-type_0_2.jpg'

img = cv.imread(picPath)
grey = cv.imread(picPath, 0)

blurImage = cv.medianBlur(img, 3)

d_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
stickMask = cv.erode(blurImage, d_kernel)
d_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
stickMask = cv.dilate(blurImage, d_kernel)


edge = cv.Canny(stickMask, 0, 100)
edge_grey = cv.Canny(grey, 0, 100)

resize_image(img, "normal", 0.4)
resize_image(stickMask, "nostick", 0.4)
resize_image(stickMask, "nostick", 0.4)
resize_image(edge, "edge color", 0.4)
resize_image(edge_grey, "canny_grey", 0.4)
cv.waitKey(0)