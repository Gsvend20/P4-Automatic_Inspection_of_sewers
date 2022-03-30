from __future__ import print_function
import cv2 as cv
import pathlib
import numpy as np

# --- To stop program press q ---

# Select picture path from .../P4-Automatic_Inspection_of_sewers/ directory
picPath = '/materials/GR-type_0_2.jpg'

# Window name options
window_capture_name = 'Original Image'
window_slider_name = 'HSV Sliders'
window_detection_name = 'Object Detection'
window_stick_name = 'Stick detection'
window_circles_name = 'Enclosing circles'
window_rect_name = 'Enclosing squares'

# HSV scale settings
max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)


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


def draw_bounding_rect(contour, image_name):
    x, y, w, h = cv.boundingRect(contour)
    cv.imshow(window_circles_name, cv.rectangle(image_name, (x, y), (x + w, y + h), (255, 0, 0), 1))
    return w, h


def draw_enclosing_circles(contour, image_name):
    # Draw enclosing circles
    (x, y), radius = cv.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv.imshow(window_rect_name, cv.circle(image_name, center, radius, (0, 255, 0), 1))
    return radius


# Create windows
cv.namedWindow(window_capture_name)
cv.namedWindow(window_slider_name)
cv.namedWindow(window_detection_name)
#cv.namedWindow(window_stick_name)
#cv.namedWindow(window_circles_name)

# Create trackbars and buttons
cv.createTrackbar(low_H_name, window_slider_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_slider_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_slider_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_slider_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_slider_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_slider_name, high_V, max_value, on_high_V_thresh_trackbar)

# Read picture
picPathFromParent = pathlib.Path().cwd().parent.as_posix() + picPath
img = cv.imread(picPathFromParent, cv.COLOR_BGR2HSV)
cv.imshow(window_capture_name, img)

# Prepare image
frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

cnt = []

# Generate mask of the stick to allow for removal from detection
upperLimit = np.array([180, 73, 255])
lowerLimit = np.array([0, 0, 136])
stickMask = cv.inRange(frame_HSV, lowerLimit, upperLimit)
# cv.imshow(window_stick_name, stickMask)


while True:
    # Threshold according to HSV sliders
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    # Combine threshold image with the stick mask and invert to only show holes
    imageBin = cv.add(frame_threshold, stickMask)
    threshInv = cv.bitwise_not(imageBin)

    # Remove unwanted noise
    strucElem = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    # strucElem = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    closedImage = close_image(threshInv, strucElem, strucElem)
    openImage = open_image(closedImage, strucElem, strucElem)

    # Show thresholded image
    cv.imshow(window_detection_name, openImage)

    # Detect contours and generate moments
    contours, hierarchy = cv.findContours(openImage, 1, 2)

    # Draw enclosing circles and bounding rectangle for all contours
    if len(contours) != 0:
        circleImg = img.copy()
        rectImg = img.copy()
        for cnt in contours:
            draw_enclosing_circles(cnt, circleImg)
            draw_bounding_rect(cnt, rectImg)

    key = cv.waitKey(30)
    if key == ord('q'):
        break

