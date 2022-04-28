import cv2 as cv
from os.path import exists
from os import mkdir, listdir

def resizenation(image, percentage):
    scale_percent = percentage  # percent of original size #this is just to make the program faster
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)

types = "down_stream"
clas = "Branch_training"
files_name = "branch"
trainingname = "branchtrain"

path = "%s/%s/%s" % (clas, clas, types)
listofvids = listdir(path)
file_number = 0
for i in range(int(len(listofvids)/3)):
    print(f"{clas}/{clas}/{types}/{i+1}_bgr.avi")
    bgrvideo = f"{clas}/{clas}/{types}/{i+1}_bgr.avi"
    irvideo = f"{clas}/{clas}/{types}/{i+1}_ir.avi"
    depthvideo = f"{clas}/{clas}/{types}/{i+1}_depth.avi"
    if exists(bgrvideo) and exists(irvideo) and exists(depthvideo):
        if not exists("./%s" % trainingname):
            mkdir("./%s" % trainingname)
        bgrcap = cv.VideoCapture(bgrvideo)
        ircap = cv.VideoCapture(irvideo)
        depthcap = cv.VideoCapture(depthvideo)
        while bgrcap.isOpened():
            ret, frame = bgrcap.read()
            ret1, irframe = ircap.read()
            ret2, depthframe = depthcap.read()
            if not ret and not ret1 and not ret2:
                print("next video")
                break
            resized = resizenation(frame, 20)
            cv.imshow("BGR image", resized)
            key = cv.waitKey(10)
            if key == ord("s"):
                print(file_number)
                bgrfilename = "%s/%s_%s.png" % (trainingname, "bgr_"+files_name, file_number)
                irfilename = "%s/%s_%s.png" % (trainingname, "ir_"+files_name, file_number)
                depthfilename = "%s/%s_%s.png" % (trainingname, "depth_" + files_name, file_number)
                cv.imwrite(bgrfilename, frame)
                if irframe == 0:
                    cv.imwrite(irfilename, irframe)
                    cv.imwrite(depthfilename, depthframe)
                file_number = file_number+1
