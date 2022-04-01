import cv2 as cv
from pathlib import Path
from collections import deque
import numpy as np

def close_image(inputimage, e_kernel, d_kernel):
    e_out = cv.erode(inputimage, e_kernel)
    d_out = cv.dilate(e_out, d_kernel)
    return d_out


def open_image(inputimage, e_kernel, d_kernel):
    d_out = cv.dilate(inputimage, d_kernel)
    e_out = cv.erode(d_out, e_kernel)
    return e_out


vid = cv.VideoCapture(Path.cwd().parent.as_posix()+'/materials/Video_long.mpg')

while (vid.isOpened()):
    # read frame
    ret, frame = vid.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # subtract with reference frame
    hsvImage = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    blurImage = cv.medianBlur(hsvImage, 3)

    upperLimit2 = np.array([60, 222, 255])
    lowerLimit2 = np.array([5, 70, 50])

    nonMask = cv.inRange(blurImage, lowerLimit2, upperLimit2)

    # Find stick
    upperLimit2 = np.array([180, 73, 255])
    lowerLimit2 = np.array([0, 0, 136])

    stickMask = cv.inRange(blurImage, lowerLimit2, upperLimit2)

    imageBin = stickMask + nonMask

    strucElem = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))

    openImage = open_image(imageBin, strucElem, strucElem)
    closedImage = close_image(openImage, strucElem, strucElem)
    threshImage = np.zeros(closedImage.shape, dtype=np.uint8)
    threshImage = cv.bitwise_not(closedImage)

    image_copy = frame.copy()

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv.findContours(image=threshImage, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area >= 15000:
            epsilon = 0.01 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)
            cv.drawContours(image_copy, [approx], 0, (0), 3)
            # Position for writing text
            x, y = approx[0][0]

            if len(approx) == 3:
                b=b
            elif len(approx) == 4:
                b = b
            elif len(approx) == 5:
                b=b
            elif 6 < len(approx) < 15:
                cv.putText(image_copy, "Ellipse", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)
            else:
                cv.putText(image_copy, "Circle", (x, y), cv.FONT_HERSHEY_COMPLEX, 1, 0, 2)


    # draw contours on the original image
    cv.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                    lineType=cv.LINE_AA)

    # show video
    cv.imshow('frame', image_copy)
    cv.imshow('Thresh', threshImage)
    cv.imshow('Blur', blurImage)
    cv.imshow('HSV', hsvImage)
    # break if i want to stop the video
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    if cv.waitKey(1) & 0xFF == ord('w'):
        for i in range(200):
            vid.read()

vid.release()
cv.destroyAllWindows()