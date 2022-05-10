import math
import cv2
import numpy as np

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)

picNumb = (0,0)
def print_pic(winname, input_img):  # Function used under debugging for showing the different changes (Thres, Morp...)
    global picNumb
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(winname, 1 + picNumb[0] * (450 + 1), 1 + picNumb[1] * (450 + 10))
    picNumb[0] += 1
    if picNumb[0] >= math.floor(1920/450):
        picNumb[0] = 0
        picNumb[1] += 1

    returnImg = cv2.resize(input_img, (450, 450))
    cv2.imshow(winname, returnImg)
    return


def medColour(input_img, thresh_img):
    numberObj = 0
    medtotal = [0, 0, 0]
    for y, row in enumerate(input_img):
        for x, pixel in enumerate(row):
            if thresh_img[y, x] > 250:
                medtotal += input_img[y, x]
                numberObj += 1
    return [medtotal[0] / numberObj, medtotal[1] / numberObj, medtotal[2] / numberObj]


def threshold_values(input_img):
    low_values = (60, 50, 40)
    high_values = (150, 150, 170)
    return_img = cv2.inRange(input_img, low_values, high_values)
    return return_img


def sobel_values(input_img):
    edgeKernel = 101  # Kernel for Egdedetection
    sobelx = cv2.Sobel(input_img, cv2.CV_64F, 1, 0,edgeKernel)
    sobely = cv2.Sobel(input_img, cv2.CV_64F, 0, 1, edgeKernel)
    sobelMix = cv2.add(sobelx, sobely)
    return sobelMix


def canny_values(input_img):
    edgeLow = 100
    edgeHigh = 200
    cannyimg = cv2.Canny(input_img, edgeLow, edgeHigh)

    return cannyimg


def blur_values(input_img, blurType):
    kern = 7  # for Bluring the image
    if blurType == 'Gaussian':
        return cv2.GaussianBlur(input_img, (kern, kern), cv2.BORDER_DEFAULT)
    if blurType == 'Median':
        return cv2.medianBlur(input_img, kern, cv2.BORDER_DEFAULT)
    if blurType == 'Blur':
        return cv2.blur(input_img, (kern, kern), cv2.BORDER_DEFAULT)


def open_img(input_img, kernelErode, kernelDila):
    # out_img = np.zeros(input_img)
    kernelErode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelErode, kernelErode))
    kernelDila = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelDila, kernelDila))
    Erodeimg = cv2.erode(input_img, kernelErode)
    return cv2.dilate(Erodeimg, kernelDila)


def close_img(input_img, kernelErode, kernelDila):
    # out_img = np.zeros(input_img)
    kernelErode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelErode, kernelErode))
    kernelDila = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelDila, kernelDila))
    Dilaimg = cv2.dilate(input_img, kernelDila)
    return cv2.erode(Dilaimg, kernelErode)


# A function for reversing the 16bit to 8bit changes
def convert_to_16(img):
    if img.shape[2] == 3:
      depth_hi_bytes, depth_lo_bytes, empty = cv2.split(img)
      return depth_lo_bytes.astype('uint16') + np.left_shift(depth_hi_bytes.astype('uint16'), 8)
    else:
      print('Image ' + ' is not 3 channel')
      return img


# This is just for trackbar callbacks
def _nothing(x):
    pass


def retrieve_trackbar(trackbar_name,img_window_name, odd_only=False):
    track_num = cv2.getTrackbarPos(trackbar_name, img_window_name)
    if not odd_only:
        return track_num
    else:
        if track_num == 0:
            return track_num
        else:
            count = track_num % 2
            if count == 0:
                track_num += 1
                return track_num
            else:
                return track_num


def define_trackbar(trackbar_name, img_window_name, max_min_values):
    cv2.namedWindow(img_window_name)
    cv2.resizeWindow(img_window_name, 200, 400);
    min_val, max_val = max_min_values
    cv2.createTrackbar(trackbar_name, img_window_name, min_val, max_val, _nothing)


def depth_to_display(depth_frame):
    depth_segmentation_value = 256  # maximum value for each channel

    # scale depth frame to fit within 3 channels of bit depth 8
    depth_frame = depth_frame / 8192 * 3 * depth_segmentation_value

    # segment depth image into 3 color channels for better visualisation
    depth_frame_b = np.where(depth_frame > 2 * depth_segmentation_value - 1,
                             cv2.subtract(depth_frame, 2 * depth_segmentation_value), np.zeros_like(depth_frame))
    depth_frame = np.where(depth_frame > 2 * depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)
    depth_frame_g = np.where(depth_frame > depth_segmentation_value - 1,
                             cv2.subtract(depth_frame, depth_segmentation_value), np.zeros_like(depth_frame))
    depth_frame_r = np.where(depth_frame > depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)

    # Aligned and depth images have different shapes, so we check for both
    shape = depth_frame_b.shape
    if len(shape) <= 1:
        depth_frame_color = cv2.merge([depth_frame_b[:, :, 0], depth_frame_g[:, :, 0], depth_frame_r[:, :, 0]])
    else:
        depth_frame_color = cv2.merge([depth_frame_b[:, :], depth_frame_g[:, :], depth_frame_r[:, :]])

    depth_frame_color = depth_frame_color.astype(np.uint8)
    return depth_frame_color