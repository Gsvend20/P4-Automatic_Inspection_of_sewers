import cv2
import numpy as np
from Functions.Featurespace import find_annodir
from Functions import imgproc_func as imf
import glob

# TODO: FIX FS thresholding, GR detection

"""
SÆT DIN PATH TIL DIT ONE DRIVE HER -> DOWNLOAD ANNOTATIONS MAPPEN FØRST
"""
# Path to folder containing the different classes
path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Training data\annotations'



# Find what classes have been found
class_name, anotations = find_annodir(path)

# Define
trackers = ['hue_upper', 'hue_lower', 'light_upper', 'light_lower', 'saturation_upper', 'saturation_lower']

hls_values = [255, 70, 255, 37, 255, 30]

blue_values = [124, 84, 119, 37, 148, 61]

scratches_values = [129, 70, 103, 21, 59, 32]

roots_values = [200, 105, 121, 101, 152, 114]

for i, tracks in enumerate(trackers):
    imf.define_trackbar(tracks, 'Base', (hls_values[i], 255))
    imf.define_trackbar(tracks, 'Cloth', (blue_values[i], 255))
    imf.define_trackbar(tracks, 'Scratches', (scratches_values[i], 255))
    imf.define_trackbar(tracks, 'ROE', (roots_values[i], 255))

imf.define_trackbar('gaussian blur', 'processing', (0,1))
# imf.define_trackbar('kernel', 'processing', (3,21))
# imf.define_trackbar('low edge', 'processing', (3,100))
# imf.define_trackbar('high edge', 'processing', (3,100))
# imf.define_trackbar('edge color space', 'processing', (0,3))

for category in class_name:
    # D used to skip categories
    D = 0
    depth_paths = glob.glob(path.replace('\\', '/') + '/' + category + '/**/*aligned*.png', recursive=True)
    for i in range(10,20):

        if D:
            break

        depth_path = depth_paths[i]
        bgr_path = depth_path.replace('aligned', 'bgr')
        depth2_path = depth_path.replace('aligned', 'depth')

        depth2_img = imf.convert_to_16(cv2.imread(depth2_path))
        depth_img = imf.convert_to_16(cv2.imread(depth_path))
        bgr_img = cv2.imread(bgr_path)
        while (True):
            kernel = imf.retrieve_trackbar('kernel', 'blurs', True)
            if imf.retrieve_trackbar('gaussian blur', 'blurs'):
                blur = cv2.GaussianBlur(bgr_img, (kernel,kernel), cv2.BORDER_DEFAULT)
            else:
                blur = cv2.medianBlur(bgr_img, kernel)

            frame_hsi = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
            frame_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)


            hls_up = []
            hls_low = []
            blue_up = []
            blue_low = []
            scr_up = []
            scr_low = []
            roe_up = []
            roe_low = []

            for i in range(0,len(trackers),2):
                hls_up.append(imf.retrieve_trackbar(trackers[i], 'Base'))
                hls_low.append(imf.retrieve_trackbar(trackers[i+1], 'Base'))
                blue_up.append(imf.retrieve_trackbar(trackers[i], 'Cloth'))
                blue_low.append(imf.retrieve_trackbar(trackers[i+1], 'Cloth'))
                scr_up.append(imf.retrieve_trackbar(trackers[i], 'Scratches'))
                scr_low.append(imf.retrieve_trackbar(trackers[i+1], 'Scratches'))
                roe_up.append(imf.retrieve_trackbar(trackers[i], 'ROE'))
                roe_low.append(imf.retrieve_trackbar(trackers[i + 1], 'ROE'))

            # Generate area of interest from pipe depth data
            aoi_end = cv2.inRange(depth_img, int(np.max(depth_img) - 100), int(np.max(depth_img)))
            aoi_pipe = cv2.inRange(depth_img, 600, int(np.max(depth_img) - 100))
            cnt, hir = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pipe_mask = np.zeros_like(depth_img).astype('uint8')
            pipe_mask = cv2.fillPoly(pipe_mask, cnt, 255)

            bg_mask = cv2.subtract(pipe_mask, aoi_end)
            bg_mask = imf.open_img(bg_mask, 21, 21)
            bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
            hsi_aoi = cv2.bitwise_and(frame_hsi, frame_hsi, mask=bg_mask)

            # Edge detection
            # edge_space = imf.retrieve_trackbar('edge color space', 'processing')
            # if edge_space == 0:
            #     canny = cv2.Canny(frame_hsi[:, :, 0], imf.retrieve_trackbar('low edge', 'processing'), imf.retrieve_trackbar('high edge', 'processing'))
            #     canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            # elif edge_space == 1:
            #     canny = cv2.Canny(frame_hsi[:, :, 1], imf.retrieve_trackbar('low edge', 'processing'),
            #                       imf.retrieve_trackbar('high edge', 'processing'))
            #     canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            # elif edge_space == 2:
            #     canny = cv2.Canny(frame_hsi[:, :, 2], imf.retrieve_trackbar('low edge', 'processing'),
            #                       imf.retrieve_trackbar('high edge', 'processing'))
            #     canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            # elif edge_space == 3:
            #     canny = cv2.Canny(imf.depth_to_display(depth_img), imf.retrieve_trackbar('low edge', 'processing'),
            #                       imf.retrieve_trackbar('high edge', 'processing'))
            #     canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)))

            """
            HER ER MASKS
            mask1 = base
            mask2 = cloth
            mask3 = scratches
            Hvis du vil have dem individuelt kan du ændre til "bin = open_img((din mask), 7,7)"
            ellers kan du udkommentere subtract delene indtil det du gerne vil have
            """

            mask1 = cv2.inRange(frame_hsi, np.asarray(hls_low),  np.asarray(hls_up))    # Threshold around highlights
            mask2 = cv2.inRange(frame_hsi,  np.asarray(blue_low),  np.asarray(blue_up))  # Remove blue, due to the piece of cloth
            mask3 = cv2.inRange(frame_hsi,  np.asarray(scr_low),  np.asarray(scr_up))  # Remove blue, due to scratches
            mask4 = cv2.inRange(frame_hsv, np.asarray(roe_low), np.asarray(roe_up))  # Find roots and pipe edges

            hsi_thresh = cv2.add(mask1, mask4)
            hsi_thresh = cv2.subtract(hsi_thresh,mask2)
            hsi_thresh = cv2.subtract(hsi_thresh, mask3)
            # hsi_thresh = cv2.add(hsi_thresh, canny)
            bin = imf.open_img(hsi_thresh, 7, 7)


            imf.resize_image(bgr_img, 'original', 0.4)
            imf.resize_image(bin.copy(), 'binary', 0.4)
            imf.resize_image(mask4, 'blur', 0.4)
            imf.resize_image(imf.depth_to_display(depth_img), 'depth', 0.4)
            # imf.resize_image(imf.depth_to_display(canny), 'canny', 0.4)
            cv2.imwrite('result.png', bin)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('d'):
                D = 1
                break