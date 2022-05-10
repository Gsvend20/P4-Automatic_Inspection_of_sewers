import cv2
import numpy as np
from Functions import imgproc_func as fnc


img = np.zeros((300,512,3), np.uint8)

fnc.define_trackbar('R', 'colors', (0, 255))
fnc.define_trackbar('G', 'colors', (0, 255))
fnc.define_trackbar('B', 'colors', (0, 255))

while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
        # get current positions of four trackbars
    r = fnc.retrieve_trackbar('R','colors')
    g = fnc.retrieve_trackbar('G', 'colors')
    b = fnc.retrieve_trackbar('B', 'colors')

    img[:] = [b, g, r]