import cv2
import numpy as np
from Functions.Featurespace import find_annodir
from Functions import imgproc_func as imf
import glob

# Path to folder containing the different classes
#path = r'C:\Users\mikip\Desktop\annotations'
path = r'C:\Users\mikip\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\annotations'

# Find what classes have been found
class_name, anotations = find_annodir(path)

# Define
trackers = ['hue_upper', 'hue_lower', 'light_upper', 'light_lower', 'saturation_upper', 'saturation_lower']
for tracks in trackers:
    imf.define_trackbar(tracks, 'trackbars 1', (0, 255))
    imf.define_trackbar(tracks, 'trackbars 2', (0, 255))
trackbar_data = [0]*len(trackers)*2

trackers_2 = ['x_dif', 'y_dif', 'distance']
for tracks in trackers_2:
    imf.define_trackbar(tracks, 'trackbars 3', (20, 1000))

for category in class_name:
    # D used to skip categories
    D = 0
    depth_paths = glob.glob(path.replace('\\', '/') + '/' + category + '/**/*aligned*.png', recursive=True)
    for i in range(10, 20):
        if D:
            break


        depth_path = depth_paths[i]
        bgr_path = depth_path.replace('aligned', 'bgr')
        depth2_path = depth_path.replace('aligned', 'depth')

        depth2_img = imf.convert_to_16(cv2.imread(depth2_path))
        depth_img = imf.convert_to_16(cv2.imread(depth_path))
        bgr_img = cv2.imread(bgr_path)

        blur = cv2.medianBlur(bgr_img, 7)

        frame_hsi = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        while (True):
            for i, tracks in enumerate(trackers):
                trackbar_data[i] = imf.retrieve_trackbar(tracks, 'trackbars 1')
                trackbar_data[i+len(trackers)] = imf.retrieve_trackbar(tracks, 'trackbars 2')
            x_diff = imf.retrieve_trackbar(trackers_2[0], 'trackbars 3')
            y_diff = imf.retrieve_trackbar(trackers_2[1], 'trackbars 3')
            distance = imf.retrieve_trackbar(trackers_2[2], 'trackbars 3')


            # Generate area of interest from pipe depth data
            aoi_end = cv2.inRange(depth_img, int(np.max(depth_img) - 100), int(np.max(depth_img)))
            aoi_pipe = cv2.inRange(depth_img, 600, int(np.max(depth_img) - 100))
            cnt, hir = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pipe_mask = np.zeros_like(depth_img).astype('uint8')
            pipe_mask = cv2.fillPoly(pipe_mask, cnt, 255)

            bg_mask = cv2.subtract(pipe_mask, aoi_end)
            imf.resize_image(bg_mask, 'pipe', 0.4)
            bg_mask = imf.open_img(bg_mask, 21, 21)
            bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
            hsi_aoi = cv2.bitwise_and(frame_hsi, frame_hsi, mask=bg_mask)

            hls_uppervalues = [trackbar_data[0], trackbar_data[2], trackbar_data[4]]
            hls_lowervalues = [trackbar_data[1], trackbar_data[3], trackbar_data[5]]
            blue_uppervalues = [trackbar_data[6], trackbar_data[8], trackbar_data[10]]
            blue_lowervalues = [trackbar_data[7], trackbar_data[9], trackbar_data[11]]

            upperLimit1 = np.array(hls_uppervalues)
            lowerLimit1 = np.array(hls_lowervalues)  # Threshold around highlights

            upperLimit2 = np.array(blue_uppervalues)
            lowerLimit2 = np.array(blue_lowervalues)  # Remove blue, due to the piece of cloth

            mask1 = cv2.inRange(frame_hsi, lowerLimit1, upperLimit1)
            mask2 = cv2.inRange(frame_hsi, lowerLimit2, upperLimit2)

            hsi_thresh = cv2.subtract(mask1,mask2)
            bin = imf.open_img(hsi_thresh, 3, 3)

            imf.resize_image(bgr_img, 'original', 0.4)
            imf.resize_image(bin.copy(), 'binary', 0.4)
            imf.resize_image(hsi_aoi.copy(), 'hsi', 0.4)
            imf.resize_image(imf.depth_to_display(depth_img), 'depth', 0.4)
            imf.resize_image(imf.depth_to_display(depth2_img), 'depth2', 1)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('d'):
                D = 1
                break