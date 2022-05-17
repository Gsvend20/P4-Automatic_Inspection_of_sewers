import cv2 as cv2
import numpy as np
import time
import math
import os.path as os


def PrintPic(winname, input_img):  # Function used under debugging for showing the different changes (Thres, Morp...)
    ResizeSize1 = input_img.shape[0]
    ResizeSize2 = input_img.shape[1]
    global PicNumb1
    global PicNumb2
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(winname, 1 + PicNumb1 * (ResizeSize1 + 1-70), 1 + PicNumb2 * (ResizeSize2 + 10))
    PicNumb1 += 1
    if PicNumb1 >= math.floor(1920/ResizeSize1):
        PicNumb1 = 0
        PicNumb2 += 1
    cv2.imshow(winname, input_img)
    return


def BLOB(input_image, Connectivity):
    if Connectivity != 4 or Connectivity != 8:
        Connectivity = 4
    output_Image = np.zeros_like(input_image)
    Blob_ID = 1
    Positions = []

    for y, row in enumerate(input_image):
        for x, pixel in enumerate(row):
            if input_image[y, x] == 255 and output_Image[y, x] == 0:
                output_Image = np.add(output_Image, GrassFire(input_image, y, x, Blob_ID, Connectivity))
                Positions.append([Blob_ID, y, x])
                Blob_ID += 1
    return output_Image, Positions


def GrassFire(input_image, y, x, Blob_ID, Connectivity):
    Changed_Image = np.zeros_like(input_image)
    CurrentPos = [y, x]
    listBlob = [CurrentPos]
    firepos = [-1, 0, 0, -1, 1, 0, 0, 1]
    if Connectivity == 4:
        while len(listBlob) > 0:
            CurrentPos = listBlob.pop(0)
            if Changed_Image[CurrentPos[0], CurrentPos[1]] == 0:
                Changed_Image[CurrentPos[0], CurrentPos[1]] = Blob_ID
                for j in range(math.floor(len(firepos)/2)):
                    NextPos = [CurrentPos[0] - ((firepos[2*j])+0), CurrentPos[1] - (firepos[2*j + 1])]
                    if NextPos[0] < input_image.shape[0] and NextPos[1] < input_image.shape[1]:
                        if input_image[NextPos[0], NextPos[1]] == 255 and Changed_Image[NextPos[0], NextPos[1]] == 0:
                            listBlob.append(NextPos)
    return Changed_Image


BlurVal = 5
Running = True
Runnings = 0
GrassTotalTime = 0
FindCountTotalTime = 0
FullProgTotalTime = 0
SetupTotalTime = 0
FileType = 'png'
FileTypeNumb = 255
BLOBPositions = []
Path = ''
# Printing images!
while Runnings <= 990:
    print(f'StartRunning filetype : {Runnings} images so far!')
    for i in range(1, FileTypeNumb):
        #if os.exists(f'{Path}00_bgr_AF_{i}.png'):
        if os.exists(f'{Path}TestImg ({i}).png'):
            print(f'Loop... image nr. {i}, ({Runnings})')
            inputimg = cv2.imread(f'{Path}TestImg ({i}).png', cv2.COLOR_BGR2HSV)
            StartProgTime = time.time()
            StartSetupTime = time.time()
            blur = cv2.medianBlur(inputimg, 13)

            frame_hsi = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
            frame_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            hls_uppervalues = [255, 255, 255]
            hls_lowervalues = [70, 37, 30]

            blue_uppervalues = [124, 119, 148]
            blue_lowervalues = [84, 37, 61]

            scr_uppervalues = [129, 103, 59]
            scr_lowervalues = [70, 21, 32]

            roe_uppervalues = [107, 114, 255]
            roe_lowervalues = [72, 28, 150]

            mask1 = cv2.inRange(frame_hsi, np.asarray(hls_lowervalues),
                                np.asarray(hls_uppervalues))  # Threshold around highlights
            mask2 = cv2.inRange(frame_hsi, np.asarray(blue_lowervalues),
                                np.asarray(blue_uppervalues))  # Remove blue, due to the piece of cloth
            mask3 = cv2.inRange(frame_hsi, np.asarray(scr_lowervalues),
                                np.asarray(scr_uppervalues))  # Remove blue, due to scratches
            mask4 = cv2.inRange(frame_hsv, np.asarray(roe_lowervalues),
                                np.asarray(roe_uppervalues))  # Add in some dark blue for roots

            hsi_thresh = cv2.add(mask1, mask4)
            hsi_thresh = cv2.subtract(hsi_thresh, mask2)
            hsi_thresh = cv2.subtract(hsi_thresh, mask3)
            # bin = imf.open_img(hsi_thresh, 5, 5)

            # Thresimg = cv2.inRange(Blur, (60, 50, 40), (150, 150, 170))

            Runnings += 1
            PicNumb1 = 0
            PicNumb2 = 0
            # cv2.waitKey(0)
            EndSetupTime = time.time()

            # Time the grassfire function
            StartTime = time.time()
            Grassimg, BLOBPositionsnew = BLOB(hsi_thresh, 4)#.astype('uint8')
            EndTime = time.time()
            GrassTotalTime += (EndTime - StartTime)

            # Time the FindCountours function

            StartTime = time.time()
            # FindCountimg, test = cv2.findContours(hsi_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            FindCountimg, test = cv2.findContours(hsi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.waitKey(1)
            if test is not None:
                EndTime = time.time()
                FindCountTotalTime += (time.time() - StartTime)
                # print(time.time() - StartTime)
                if (EndTime-StartTime) <= 0.000000005:
                    Runnings -= 1
                    print('to fast! removes 1')
            # print(f'FindCountimg took = {(EndTime - StartTime) / 60} Seconds!')
            EndProgTime = time.time()
            FullProgTotalTime += (EndProgTime - StartProgTime)
            SetupTotalTime += (EndSetupTime - StartSetupTime)
            # print(f'GrassFire took = {(EndTime - StartTime) / 60} Seconds! (img:{Runnings})')
            Grassimg2 = (Grassimg * (250 / Grassimg.max())).astype('uint8')
            BLOBPositions.append(BLOBPositionsnew)
            PrintPic('Input Image', inputimg)
            PrintPic('Thresholded Image', hsi_thresh)
            PrintPic('GrassFire', Grassimg2)
            outPic = np.zeros_like(hsi_thresh)
            for m, cnt in enumerate(FindCountimg):
                cv2.drawContours(outPic, [cnt], 0, int(m*((200 / Grassimg.max())+50)), thickness=-1)
            PrintPic('Countour Image', outPic)
            cv2.waitKey(1)
            if Runnings <= 25:
                time.sleep(0.25)

        else:
            print('Image not Exist!')
            Running = False


print(f'Average Time for GrassFire ({Runnings} runs) = {GrassTotalTime/Runnings*1000} ms')
print(f'Average Time for FindCounturs ({Runnings} runs) = {FindCountTotalTime/Runnings*1000} ms')
print(f'Average Time for FullProg ({Runnings} runs) = {FullProgTotalTime/Runnings*1000} ms')
print('Output of some of the positions of some of the Images:')
for i in range(math.floor(len(BLOBPositions)/100)):
    print(BLOBPositions[20*i])
