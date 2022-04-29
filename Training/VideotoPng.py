import cv2 as cv
from os.path import exists
from os import mkdir, listdir
import numpy as np

def resizenation(image, percentage):
    scale_percent = percentage  # percent of original size #this is just to make the program faster
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)

folder = '2nd_training_set'
files_name = "Sand"
trainingname = "Sandtraindata"

if not exists("./%s" % trainingname):
    mkdir("./%s" % trainingname)
path = f"{folder}"
listofvids = listdir(path)
path = f"{folder}"
listofvids = listdir(path)

for j, vid in enumerate(listofvids):
    spit = vid.split("_")
    if spit[2] == 'bgr.avi':
        bgrvideo = f"{folder}/{listofvids[j]}"
        depthvideo = f"{folder}/{listofvids[j+1]}"
        irvideo = f"{folder}/{listofvids[j+2]}"
        types = spit[0]
        videonum = spit[1]
        filenum = 0
        if exists(bgrvideo) and exists(irvideo) and exists(depthvideo):
            if not exists(f"{trainingname}/{types}"):
                mkdir(f"{trainingname}/{types}")
            bgrcap = cv.VideoCapture(bgrvideo)
            ircap = cv.VideoCapture(irvideo)
            depthcap = cv.VideoCapture(depthvideo)
            while bgrcap.isOpened():
                ret, frame = bgrcap.read()
                ret1, irframe = ircap.read()
                ret2, depthframe = depthcap.read()
                if not ret:
                    print("next video")
                    break
                resized = resizenation(frame, 20)
                cv.imshow("BGR image", resized)
                key = cv.waitKey(10)
                if key == ord("q"):
                    break
                if key == ord("s"):
                    filenum = str(filenum)
                    filenum = filenum.zfill(3)
                    bgrfilename = f"{trainingname}/{types}/{filenum}_bgr_{files_name}_{videonum}.png"
                    irfilename = f"{trainingname}/{types}/{filenum}_ir_{files_name}_{videonum}.png"
                    depthfilename = f"{trainingname}/{types}/{filenum}_depth_{files_name}_{videonum}.png"
                    cv.imwrite(bgrfilename, frame)
                    filenum = int(filenum)
                    filenum = filenum+1
                    cv.imwrite(irfilename, irframe)
                    cv.imwrite(depthfilename, depthframe)
