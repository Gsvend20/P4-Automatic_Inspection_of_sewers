import cv2
import numpy as np
from Functions import imgproc_func as imf


def add_adaptive_frame(frame_depth):
    # Generate area of interest from pipe depth data, by finding the end of the pipe
    aoi_end = cv2.inRange(frame_depth, int(np.max(frame_depth) - 100), int(np.max(frame_depth)))
    # Then the front of the pipe is extracted
    aoi_pipe = cv2.inRange(frame_depth, 600, int(np.max(frame_depth) - 100))
    cnt, hir = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Make a mask out of both
    pipe_mask = np.zeros_like(frame_depth).astype('uint8')
    pipe_mask = cv2.fillPoly(pipe_mask, cnt, 255)
    bg_mask = cv2.subtract(pipe_mask, aoi_end)
    bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    # use the mask to generate an area of interest in the depth data
    fg_d_frame = cv2.bitwise_and(frame_depth, frame_depth, mask=bg_mask)
    return fg_d_frame


def get_binary(bgr_frame, depth_mask, display_result=False):
    # These are all the upper and lower bounds for the thresholding
    base_up = [255, 255, 255]
    base_low = [70, 37, 30]

    blue_up = [124, 140, 150]
    blue_low = [84, 37, 61]

    scr_up = [129, 103, 59]
    scr_low = [70, 21, 32]

    # Begin treating the image in the same way you would detect flaws
    blur = cv2.GaussianBlur(bgr_frame, (13, 13), cv2.BORDER_DEFAULT)

    frame_hsi = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)   # Color space conversion

    # Thresholding
    mask1 = cv2.inRange(frame_hsi, np.asarray(base_low), np.asarray(base_up))  # Threshold around highlights
    mask2 = cv2.inRange(frame_hsi, np.asarray(blue_low), np.asarray(blue_up))  # Remove blue, due to the piece of cloth
    mask3 = cv2.inRange(frame_hsi, np.asarray(scr_low), np.asarray(scr_up))  # Remove blue, due to scratches

    hsi_thresh = cv2.add(mask1, depth_mask)
    hsi_thresh = cv2.subtract(hsi_thresh, mask2)
    hsi_thresh = cv2.subtract(hsi_thresh, mask3)

    binary = imf.open_img(hsi_thresh, 5, 5)  # By opening we remove noise and the edges that are of no interest

    if display_result:
        imf.resize_image(binary, 'Binary image', 0.5)
    return binary
