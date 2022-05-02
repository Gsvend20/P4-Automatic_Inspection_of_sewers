import cv2 as cv
import numpy as np
from os.path import exists
from os import mkdir, listdir

def resizenation(image, percentage):
    scale_percent = percentage  # percent of original size #this is just to make the program faster
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)
#folder where the training videos are
folder = '2nd training set'
#name of the injury
files_name = "ROE"
#name of the folder where the training images will land
trainingname = "roots_traindata2"

#creates a folder for all the training data
if not exists("./%s" % trainingname):
    mkdir("./%s" % trainingname)
#path of the folder of your choice
path = f"{folder}"
listofvids = listdir(path)

# loops through all the videos and shows them to you, if s is pressed image is saved
for j, vid in enumerate(listofvids):
    # Puts out a string type of the name of the path to the three types of video.
    numb = 0
    spit = vid.split("_")
    if spit[2] == 'bgr.avi':
        bgrvideo = f"{folder}/{listofvids[j]}"
        depthvideo = f"{folder}/{listofvids[j+1]}"
        irvideo = f"{folder}/{listofvids[j+2]}"
        types = spit[0]
        videonum = spit[1]
        filenum = 0

        # If the path to the three videos exists create a sub folder of the damage type and then subfolders for
        # the three video types
        if exists(bgrvideo) and exists(irvideo) and exists(depthvideo):
            if not exists(f"{trainingname}/{types}"):
                mkdir(f"{trainingname}/{types}")
                mkdir(f"{trainingname}/{types}/bgr")
                mkdir(f"{trainingname}/{types}/ir")
                mkdir(f"{trainingname}/{types}/depth")
            #reads the frames of the three videos
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
                key = cv.waitKey(25)
                if key == ord("q"):
                    break
                if key == ord("s"):
                    filenum = str(filenum)
                    filenum = filenum.zfill(3)
                    bgrfilename = f"{trainingname}/{types}/bgr/{filenum}_bgr_{files_name}_{videonum}.png"
                    irfilename = f"{trainingname}/{types}/ir/{filenum}_ir_{files_name}_{videonum}.png"
                    depthfilename = f"{trainingname}/{types}/depth/{filenum}_depth_{files_name}_{videonum}.png"
                    cv.imwrite(bgrfilename, frame)
                    filenum = int(filenum)
                    filenum = filenum+1
                    ir_hi_bytes, ir_lo_bytes, empty = cv.split(irframe)
                    irframe = ir_lo_bytes.astype('uint16') + np.left_shift(ir_hi_bytes.astype('uint16'), 8)
                    cv.imwrite(irfilename, irframe)
                    depth_hi_bytes, depth_lo_bytes, empty = cv.split(depthframe)
                    depthframe = depth_lo_bytes.astype('uint16') + np.left_shift(depth_hi_bytes.astype('uint16'), 8)
                    cv.imwrite(depthfilename, depthframe)
                    numb += 1
                    print(numb)
