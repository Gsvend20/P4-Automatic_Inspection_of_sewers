import cv2
import numpy as np
from os import listdir

depth_folder = 'roots_traindata/20/depth/'
ir_folder = 'roots_traindata/20/ir'

listofvids = listdir(depth_folder)

# loops through all the videos
for j, vid in enumerate(listofvids):
    depth_path = f"{depth_folder}/{listofvids[j]}"

    # Fetch depth image and merge its channels to form 16-bit image
    depth_img = cv2.imread(depth_path)
    if depth_img.shape[2] == 3:
        depth_hi_bytes, depth_lo_bytes, empty = cv2.split(depth_img)
        depth_img = depth_lo_bytes.astype('uint16') + np.left_shift(depth_hi_bytes.astype('uint16'), 8)
        print(depth_img.dtype)
        cv2.imshow('depth image', depth_img)
        cv2.waitKey(0)
        #cv2.imwrite(depth_path, depth_img)
    else:
        print('Image ' + depth_path + ' is not 3 channel')

listofvids = listdir(ir_folder)

# loops through all the videos
for j, vid in enumerate(listofvids):
    ir_path = f"{ir_folder}/{listofvids[j]}"
    # Fetch ir image and merge its channels to form 16-bit image
    ir_img = cv2.imread(ir_path)
    if ir_img.shape[2] == 3:
        ir_hi_bytes, ir_lo_bytes, empty = cv2.split(ir_img)
        ir_img = ir_lo_bytes.astype('uint16') + np.left_shift(ir_hi_bytes.astype('uint16'), 8)
        print(ir_img.dtype)
        #cv2.imwrite(ir_path, ir_img)
    else:
        print('Image ' + ir_path + ' is not 3 channel')