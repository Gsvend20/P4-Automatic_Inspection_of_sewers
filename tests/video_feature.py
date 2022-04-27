import cv2
import numpy as np
from pathlib import Path

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)


for n in range(1,6):
    v_num = n
    path_bgr = f'{Path.cwd().parent.as_posix()}/sewer recordings/Training data/Branching pipe/'

    vid = cv2.VideoCapture(path_bgr+f'{v_num}_bgr.avi')
    while vid.isOpened():
        # read frame
        ret, frame = vid.read()
        if not ret:
            break
        blur = cv2.blur(frame, (7, 7))
        hsvImg = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # first cut out the apple part
        upperLimit1 = np.array([95, 255, 230])
        lowerLimit1 = np.array([40, 0, 0])

        upperLimit2 = np.array([240, 255, 230])
        lowerLimit2 = np.array([120, 0, 0])

        mask1 = cv2.inRange(hsvImg, lowerLimit1, upperLimit1)
        mask2 = cv2.inRange(hsvImg, lowerLimit2, upperLimit2)
        thresh = mask1 + mask2
        edge = cv2.Canny(thresh, 0, 100)
        imgBin = thresh+edge

        # HSL thresholding is used due to the background always being white
        # a different method is needed if live footage was used
        resize_image(imgBin, 'image', 0.4)
        resize_image(edge, 'edge', 0.4)
        resize_image(thresh, 'thresh', 0.4)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
