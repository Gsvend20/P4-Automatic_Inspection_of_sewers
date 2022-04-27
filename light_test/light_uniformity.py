import cv2
from pathlib import Path

for n in range(1,6):
    v_num = n
    path_bgr = f'{Path.cwd().parent.as_posix()}/sewer recordings/Luminance test/'

    vid = cv2.VideoCapture(path_bgr+f'{v_num}_bgr.avi')
    mean_intensity = 0
    frame_length = 0
    max_min_intensity = []

    while vid.isOpened():
        # read frame
        ret, frame = vid.read()
        if not ret:
            break
        frame_length += 1
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mean_intensity += grey.mean()
        max_min_intensity.append(grey.max())
        max_min_intensity.append(grey.min())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(f'runnumber = {n}')
    print(f'Mean = {mean_intensity/frame_length}, min = {min(max_min_intensity)}, max = {max(max_min_intensity)}')