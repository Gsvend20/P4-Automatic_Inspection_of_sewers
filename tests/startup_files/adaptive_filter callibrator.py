import cv2 as cv
from pathlib import Path

# scale settings
max_value = 199
max_constant = 30
kernel_size = 3
constant_size = 0


def on_kernel_thresh_trackbar(val):
    global kernel_size
    global max_value
    kernel_size = val
    kernel_size = min(max_value + 1, kernel_size)
    cv.setTrackbarPos('kernel size', 'gauss', kernel_size)
    cv.setTrackbarPos('kernel size', 'median', kernel_size)


def on_constant_thresh_trackbar(val):
    global constant_size
    global max_constant
    constant_size = val
    constant_size = min(max_constant - 1, constant_size)
    cv.setTrackbarPos('constant size', 'gauss', constant_size)
    cv.setTrackbarPos('constant size', 'median', constant_size)



# read pic
number = 7
picPath = Path.cwd().parent.as_posix()+'/materials/GR-type_0_'+str(number)+'.jpg'
img = cv.imread(picPath)
grImg = cv.imread(picPath, 0)

blurImg = cv.medianBlur(grImg, 7)

# see the results
cv.namedWindow('before')
cv.namedWindow('slider')
cv.namedWindow('gaus')
cv.namedWindow('median')

cv.createTrackbar('kernel size', 'slider', kernel_size, max_value, on_kernel_thresh_trackbar)
cv.createTrackbar('constant size', 'slider', constant_size, max_constant, on_constant_thresh_trackbar)

cv.imshow('before', grImg)
frame_threshold = grImg

while True:
    if kernel_size%2 == 0:
        if kernel_size == 0 or kernel_size == 1:
            kernel_size = kernel_size
        else:
            kernel_size += 1
            frame_threshold = cv.adaptiveThreshold(blurImg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                               kernel_size, constant_size)
            frame_threshold2 = cv.adaptiveThreshold(blurImg, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                                   kernel_size, constant_size)
    else:
        if kernel_size == 0 or kernel_size == 1:
            kernel_size=kernel_size
        else:
            frame_threshold = cv.adaptiveThreshold(blurImg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                                   kernel_size, constant_size)
            frame_threshold2 = cv.adaptiveThreshold(blurImg, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                                   kernel_size, constant_size)

    cv.imshow('gauss', frame_threshold)
    cv.imshow('median', frame_threshold2)

    key = cv.waitKey(30)
    if key == ord('q'):
        break
