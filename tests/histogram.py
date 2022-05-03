import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from os import mkdir, listdir
from Functions import imgproc_func as func

path = f'{Path.cwd().parent.as_posix()}/sewer recordings/Histogram'
listofvids = listdir(path)

for vid in listofvids:
    vid = cv2.VideoCapture(f'{path}/{vid}')
    mean_intensity = 0
    frame_length = 0
    max_min_intensity = []
    while vid.isOpened():
        # read frame
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=40)
        func.resize_image(frame, 'image', 0.4)

        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('w'):

            full_frame = []
            full_frame.append(frame)
            full_frame.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
            full_frame.append(cv2.cvtColor(frame, cv2.COLOR_BGR2HLS))
            full_frame.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            color = ('blue', 'green', 'red')
            fig, axs = plt.subplots(4, figsize=(12, 6))
            fig.suptitle('bgr, hsv, hsi, grey')
            for n in range(3):
                for i, col in enumerate(color):
                    hist = cv2.calcHist([full_frame[n]], [i], None, [256], [0, 256])
                    axs[n].plot(hist, color=col)
                    plt.xlim([0, 256])
            axs[3].plot(cv2.calcHist([full_frame[3]], [0], None, [256], [0, 256]), color='black')
            plt.xlim([0, 256])
            plt.show()