# Import part
from __future__ import print_function
# from os.path import exists
import cv2 as cv
# import pathlib
from Functions import Funcs as func
import materials.datafile as Vas


# input_vid = func.GetVid('1_bgr')
input_vid = func.GetVid(r'C:\Users\Glenn\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\Roots\2nd training set\12_2_1bgr_.avi')
# Read until video is completed
while input_vid.isOpened():
    Vas.PicNumb1 = 0
    Vas.PicNumb2 = 0
    # input_vidture frame-by-frame
    ret, frame = input_vid.read()
    if ret:
        Blur_img = func.Blur(frame, 'Gaussian')
        Thres_img = func.Threshold(Blur_img)
        Canny_img = func.Canny(Thres_img)
        Sobel_img = func.Sobel(Thres_img)
        # func.PrintPic('vid', frame)
        func.PrintPic('Blur', Blur_img)
        func.PrintPic('Threshold', Thres_img)
        func.PrintPic('Egde_img Canny', Canny_img)
        func.PrintPic('Egde_img Sobel', Sobel_img)
        # cv.imshow('Frame', frame)

        if cv.waitKey(25) & 0xFF == ord('n'):
            while not cv.waitKey(25) & 0xFF == ord('m'):
                {}

        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

        # Break the loop
    else:
        print("Can't receive frame (stream end?). Exiting ...")
        break

# When everything done, release the video input_vidture object
input_vid.release()

# Closes all the frames
cv.destroyAllWindows()
