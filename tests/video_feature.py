import cv2
import numpy as np
from pathlib import Path

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)


for n in range(1,4):
    v_num = n
    path_bgr = f'{Path.cwd().parent.as_posix()}/sewer recordings/'
    #/sewer recordings/conversion files data/Offset pipe/Along pipe/30mm/
    vid = cv2.VideoCapture(path_bgr+f'{v_num}_bgr.avi')
    while vid.isOpened():
        # read frame
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=40)
        blur = cv2.blur(frame, (3, 3))
        hsvImg = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        thresh = cv2.inRange(hsvImg, np.array([45, 0, 0]), np.array([255, 255, 255]))
        imgBin = thresh
        rows = imgBin.shape[0]
        circles = cv2.HoughCircles(imgBin, cv2.HOUGH_GRADIENT, 1, rows / 16,
                                  param1=50, param2=20,
                                  minRadius=400, maxRadius=500)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(imgBin, center, 1, 1, 3)
                # circle outline
                radius = i[2]
                cv2.circle(imgBin, center, radius, (255, 0, 255), 3)

        # HSL thresholding is used due to the background always being white
        # a different method is needed if live footage was used
        resize_image(frame, 'image', 0.4)
        #resize_image(edge, 'edge', 0.4)
        resize_image(thresh, 'thresh', 0.4)
        resize_image(imgBin, 'final', 0.4)

        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('p'):
            cv2.waitKey(25)
            cv2.waitKey(0)
