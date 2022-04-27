# Import part
from __future__ import print_function
from os.path import exists
import cv2 as cv
import pathlib
from Functions import Funcs as func
import materials.datafile as Vas


input_vid = func.GetVid('1_bgr')
# Read until video is completed
while input_vid.isOpened():
    Vas.PicNumb1 = 0
    Vas.PicNumb2 = 0
    # input_vidture frame-by-frame
    ret, frame = input_vid.read()
    if ret:
        func.PrintPic('vid', frame)
        Thres_img = func.Threshold(frame)
        func.PrintPic('Threshold', Thres_img)
        # cv.imshow('Frame', frame)

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
