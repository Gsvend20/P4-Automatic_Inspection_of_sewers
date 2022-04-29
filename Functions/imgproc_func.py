import math
import cv2

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