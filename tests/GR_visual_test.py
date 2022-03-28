import cv2 as cv
from pathlib import Path
import numpy as np

def close_image(inputimage, e_kernel, d_kernel):
    e_out = cv.erode(inputimage, e_kernel)
    d_out = cv.dilate(e_out, d_kernel)
    return d_out


def open_image(inputimage, e_kernel, d_kernel):
    d_out = cv.dilate(inputimage, d_kernel)
    e_out = cv.erode(d_out, e_kernel)
    return e_out

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv.namedWindow(image_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(image_name, int(width), int(height))
    cv.imshow(image_name, image)


# read pic
number = 4
picPath = Path.cwd().parent.as_posix()+'/materials/GR-type_0_'+str(number)+'.jpg'
image = cv.imread(picPath)
grPic = cv.imread(picPath, 0)

hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
blurImage = cv.medianBlur(hsvImage, 3)

upperLimit2 = np.array([60, 222, 255])
lowerLimit2 = np.array([5, 70, 90])

nonMask = cv.inRange(blurImage, lowerLimit2, upperLimit2)

#Find stick
upperLimit2 = np.array([180, 73, 255])
lowerLimit2 = np.array([0, 0, 136])

stickMask = cv.inRange(blurImage, lowerLimit2, upperLimit2)

imageBin = stickMask+nonMask

strucElem = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))

openImage = open_image(imageBin, strucElem, strucElem)
closedImage = close_image(openImage, strucElem, strucElem)
threshImage = np.zeros(closedImage.shape, dtype=np.uint8)
threshImage = cv.bitwise_not(closedImage)

image_copy = image.copy()

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv.findContours(image=threshImage, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

# data from contours
for cnt in contours:
    area = cv.contourArea(cnt)
    if area >= 20000:
        circle_c = cnt
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv.circle(image_copy, center, radius, (255, 0, 0), 2)



# draw contours on the original image
cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 lineType=cv.LINE_AA)

# see the results
resize_image(imageBin, 'Binary_image', 0.4)
resize_image(openImage, 'opened image', 0.4)
resize_image(threshImage, 'Thresh image', 0.4)
resize_image(image_copy, 'Final image', 0.4)
cv.waitKey(0)
