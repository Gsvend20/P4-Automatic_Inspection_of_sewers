import cv2
import numpy as np
from Functions import kinect_python_functions as map_kin
import glob

""""
DEBUGER
"""
DEBUG = 0

def convert_to_color(depth_frame):
    depth_segmentation_value = 256  # maximum value for each channel

    # scale depth frame to fit within 3 channels of bit depth 8
    depth_frame = depth_frame / 8192 * 3 * depth_segmentation_value

    # segment depth image into 3 color channels for better visualisation
    depth_frame_b = np.where(depth_frame > 2 * depth_segmentation_value - 1,
                             cv2.subtract(depth_frame, 2 * depth_segmentation_value), np.zeros_like(depth_frame))
    depth_frame = np.where(depth_frame > 2 * depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)
    depth_frame_g = np.where(depth_frame > depth_segmentation_value - 1,
                             cv2.subtract(depth_frame, depth_segmentation_value), np.zeros_like(depth_frame))
    depth_frame_r = np.where(depth_frame > depth_segmentation_value - 1, np.zeros_like(depth_frame), depth_frame)
    depth_frame_color = cv2.merge([depth_frame_b[:,:,0], depth_frame_g[:,:,0], depth_frame_r[:,:,0]])
    depth_frame_color = depth_frame_color.astype(np.uint8)
    return depth_frame_color

def convert_to_16(img):
    if img.shape[2] == 3:
      depth_hi_bytes, depth_lo_bytes, empty = cv2.split(img)
      return depth_lo_bytes.astype('uint16') + np.left_shift(depth_hi_bytes.astype('uint16'), 8)
    else:
      print('Image ' + ' is not 3 channel')
      return img


def start_saving(file,path, size):
    file.open(path,
                          cv2.VideoWriter_fourcc(*"LAGS"),
                          float(30),
                          size)


def save_video(file, frame):
    rgbd_hi_bytes = np.right_shift(frame, 8).astype('uint8')
    rgbd_lo_bytes = frame.astype('uint8')
    split_rgbd_frame = cv2.merge([rgbd_hi_bytes[:,:,0], rgbd_lo_bytes[:,:,0], np.zeros_like(rgbd_hi_bytes[:,:,0])])
    file.write(split_rgbd_frame)



folder_path = ''
listof_depth = glob.glob(folder_path+'/*depth.avi')
video_writer = cv2.VideoWriter()

# loops through all the videos
for i in range(len(listof_depth)):
    depth_src = cv2.VideoCapture(listof_depth[i])
    start_saving(video_writer, folder_path + '/' + str(i) + '_aligned.avi', (1080, 1920))
    print('Saving video '+str(i+1)+'/'+str(len(listof_depth)))
    while True:
        ret, depth_frame = depth_src.read()
        if not ret:
            break

        depth = cv2.rotate(convert_to_16(depth_frame), cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth_in_color = map_kin.convert_to_depthCamera(depth)
        save_video(video_writer, cv2.rotate(depth_in_color, cv2.ROTATE_90_CLOCKWISE))


        # Show aligned frames as they are recorded
        if DEBUG:
            img = convert_to_color(depth_in_color)
            cv2.imshow('KINECT aligned image', cv2.resize(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), (int(1080/2), int(1960/2))))
            key = cv2.waitKey(30)
            if key == ord('q'):
                break


