import math

import cv2
import json
import numpy as np
import materials.datafile as Vas


def saveimg(img, errorClass):
    import cv2
    from datetime import date
    import pathlib
    cv2.imwrite(f'{pathlib.Path().cwd().parent.as_posix()}/materials/ClassedPics/{str(date.today().strftime("%d_%m"))}_{errorClass}.jpg', img)
    # cv2.imwrite(r'\P4-Grise_Projekt/materials/ClassedPics/'+str(errorClass), img)


def saveVariables(JVasr):
    JsonFile = open("Variables.json", "w")
    json.dump(JVasr, JsonFile, indent=4)
    JsonFile.close()


def GetVid(name_vid):
    import pathlib
    cap = cv2.VideoCapture(f'{pathlib.Path().cwd().parent.as_posix()}/sewer recordings/Training data/Branching pipe/{name_vid}.avi')
#    cap = cv2.VideoCapture(f'{name_vid}.avi')
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
        return False

    return_vid = cap
    return return_vid


def PrintPic(winname, input_img):  # Function used under debugging for showing the different changes (Thres, Morp...)
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(winname, 1 + Vas.PicNumb1 * (Vas.ResizeSize1 + 1), 1 + Vas.PicNumb2 * (Vas.ResizeSize2 + 10))
    Vas.PicNumb1 += 1
    if Vas.PicNumb1 >= math.floor(1920/Vas.ResizeSize1):
        Vas.PicNumb1 = 0
        Vas.PicNumb2 += 1

    returnimg = cv2.resize(input_img, (Vas.ResizeSize1, Vas.ResizeSize2))
    cv2.imshow(winname, returnimg)
    return


def MedColour(input_img, Thres_img):
    # out_img = np.zeros(input_img)    for y, row in enumerate(input_img):
    NumberObj = 0
    Medtotal = [0, 0, 0]
    for y, row in enumerate(input_img):
        for x, pixel in enumerate(row):
            if Thres_img[y, x] > 250:
                Medtotal += input_img[y, x]
                NumberObj += 1
    Medc = [Medtotal[0] / NumberObj, Medtotal[1] / NumberObj, Medtotal[2] / NumberObj]

    return Medc

def Threshold(input_img):
    return_img = cv2.inRange(input_img, (Vas.low_H, Vas.low_S, Vas.low_V), (Vas.high_H, Vas.high_S, Vas.high_V))
    return return_img


def Dilation(input_img, Kernel):
    # out_img = np.zeros(input_img)
    out_img = cv2.dilate(input_img, Kernel)

    return out_img


def Erosion(input_img, Kernel):
    # out_img = np.zeros(input_img)
    out_img = cv2.erode(input_img, Kernel)

    return out_img


def Opening(input_img, KernelErode, KernelDila):
    # out_img = np.zeros(input_img)
    KernelErode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KernelErode, KernelErode))
    KernelDila = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KernelDila, KernelDila))
    Erodeimg = Erosion(input_img, KernelErode)
    out_img = Dilation(Erodeimg, KernelDila)

    return out_img


def Closing(input_img, KernelErode, KernelDila):
    # out_img = np.zeros(input_img)
    KernelErode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KernelErode, KernelErode))
    KernelDila = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KernelDila, KernelDila))
    Dilaimg = Dilation(input_img, KernelDila)
    out_img = Erosion(Dilaimg, KernelErode)

    return out_img


def GetFeatures(input_img):
    medcolor = [0, 0, 0]
    area = 0
    perimeter = 0
    circularity = 0
    height, width, _ = input_img.shape
    # area is calculated as “height x width”
    areapic = height * width
    Vas.picNumb = 0
    Vas.PicNumb1 = 0
    Vas.PicNumb2 = 0
    frame_HSV = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    blur_image = cv2.GaussianBlur(frame_HSV, (Vas.Blur, Vas.Blur), cv2.BORDER_DEFAULT)
    frame_threshold = cv2.inRange(blur_image, (Vas.low_H, Vas.low_S, Vas.low_V), (Vas.high_H, Vas.high_S, Vas.high_V))
    Morp_image = Opening(frame_threshold, Vas.ErodeKernel, Vas.DilatKernel)
    contours, hierarchy = cv2.findContours(Morp_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conimg = input_img.copy()
    cv2.drawContours(conimg, contours, -1, (0, 255, 0), 3)
    for k in range(len(contours)):
        cont = contours[k]
        areacont = cv2.contourArea(cont)
        area = areacont / areapic
        if 0.03 < area < 0.80:
            medcolor = MedColour(input_img, Morp_image)
            perimeter = cv2.arcLength(cont, True)
            circularity = (4 * np.pi * area / (perimeter ** 2))

    data = [medcolor[0], medcolor[1], medcolor[2], area, perimeter, circularity]
    return data
