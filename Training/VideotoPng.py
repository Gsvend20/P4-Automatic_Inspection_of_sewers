import cv2 as cv
from os.path import exists
from os import mkdir, listdir

videos = 5
types = "down_stream"
clas = "Branch_training"
Vision = "bgr"
files_name = "branch"

def resizenation(image, percentage):
    scale_percent = percentage  # percent of original size #this is just to make the program faster
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv.resize(image, dim, interpolation=cv.INTER_AREA)
path = "%s/%s/%s" % (clas, clas, types)
listofvids = listdir(path)
file_number = 0
for i in range(int(len(listofvids)/3)):
    video = "%s/%s/%s/%s_%s.avi" % (clas, clas, types, i, Vision)
    if exists(video):
        cap = cv.VideoCapture(video)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("next video")
                break
            resized = resizenation(frame, 20)
            cv.imshow("BGR image", resized)
            key = cv.waitKey(10)
            if key == ord("s"):
                print(file_number)
                filename = "%s_%s.png" % (files_name, file_number)
                cv.imwrite(filename, frame)
                file_number = file_number+1
