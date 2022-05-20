import cv2
from tests import load_saved_videos

path = 'sewer recordings/2'

loader = load_saved_videos.VideoLoader()

loader.load_videos(path)

while True:
    bgr_frame, depth_frame, ir_frame, read_success = loader.get_next_frames()
    if not read_success:
        exit(1)
    else:
        cv2.imshow('color_video', bgr_frame)
        cv2.imshow('depth_video', depth_frame)
        cv2.imshow('ir_video', ir_frame)

    cv2.waitKey(10)
