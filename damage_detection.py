from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
from Functions import imgproc_func as imf
import cv2
import numpy as np
import os


def combine_contours(contours_1, contours_2, img_shape):
    intersecting_contours = []
    for cnt1 in contours_1:
        for cnt2 in contours_2:
            # Create image filled with zeros the same size of original image
            blank = np.zeros(img_shape[0:2])

            # Copy each contour into its own image and fill it with '1'
            image1 = cv2.drawContours(blank.copy(), cnt1, 0, 1)
            image2 = cv2.drawContours(blank.copy(), cnt2, 1, 1)

            # Use the logical AND operation on the two images
            # Since the two images had bitwise and applied to it,
            # there should be a '1' or 'True' where there was intersection
            # and a '0' or 'False' where it didnt intersect
            if np.logical_and(image1, image2).size > 0:
                intersecting_contours.append(cnt1)
                intersecting_contours.append(cnt2)
    return intersecting_contours


# Path to video folders
folders_path = 'Test'

# Load saved video streams
vid_bgr = cv2.VideoCapture()
vid_bgrd = cv2.VideoCapture()

# Trackbars for adjusting values
# imf.define_trackbar('Blur level', 'trackbars', (0, 31))
#imf.define_trackbar('min', 'trackbars', (500, 2000000))
#imf.define_trackbar('max', 'trackbars', (500, 2000000))
#imf.define_trackbar('Closing level', 'trackbars', (0, 255))
#imf.define_trackbar('Min depth', 'trackbars', (500, 1500))
imf.define_trackbar('Kernel', 'trackbars', (3, 30))
hls_values = [255, 70, 255, 38, 255, 24]

# load classifier
#classifier = Classifier()
#classifier.load_trained_classifier('training_data.pkl')

while True:
    vid_bgr.open(f"{folders_path}/1_bgr.avi")
    vid_bgrd.open(f"{folders_path}/1_aligned.avi")

    while vid_bgr.isOpened():
        # Fetch next frame
        ret, frame_bgr = vid_bgr.read()
        if not ret:
            break
        ret, frame_bgrd_8bit = vid_bgrd.read()
        if not ret:
            break
        # Convert depthmap to 16 bit depth
        frame_bgrd = imf.convert_to_16(frame_bgrd_8bit)

        #frame_bgrd = cv2.GaussianBlur(frame_bgrd, (5, 5), 0)
        frame_bgr = cv2.medianBlur(frame_bgr, 5)

        frame_hsi = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        grey_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        #
        # # Remove noise
        # kblur = imf.retrieve_trackbar('Blur level', 'trackbars', True)
        # if kblur >= 3:
        #     grey_frame = cv2.medianBlur(grey_frame, kblur)
        #
        # # Edge detection in grayscale
        # cmin = imf.retrieve_trackbar('Canny min', 'trackbars')
        # cmax = imf.retrieve_trackbar('Canny max', 'trackbars')
        # grey_frame = cv2.Canny(grey_frame, cmin, cmax)
        #
        # clev = imf.retrieve_trackbar('Closing level', 'trackbars', True)
        # if clev >= 3:
        #     grey_frame = imf.close_img(grey_frame, clev, clev)

        # Threshold viewing range in depth image to generate aoi
        #dmin = imf.retrieve_trackbar('Min depth', 'trackbars')
        #dmax = int(np.max(frame_bgrd) - 300)
        #aoi_near = cv2.inRange(frame_bgrd, dmin, dmax)

        #dmax = int(np.max(frame_bgrd) + imf.retrieve_trackbar('Max depth', 'trackbars'))

        # Generate area of interest from pipe depth data
        aoi_end = cv2.inRange(frame_bgrd, int(np.max(frame_bgrd) - 100), int(np.max(frame_bgrd)))
        aoi_pipe = cv2.inRange(frame_bgrd, 600, int(np.max(frame_bgrd) - 100))
        cnt, hir = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pipe_mask = np.zeros_like(frame_bgrd).astype('uint8')
        pipe_mask = cv2.fillPoly(pipe_mask, cnt, 255)
        bg_mask = cv2.subtract(pipe_mask, aoi_end)
        bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        hsi_aoi = cv2.bitwise_and(frame_hsi, frame_hsi, mask=bg_mask)

        # Create BLOB for GR from depth frames
        gr_dep_mask = cv2.bitwise_and(frame_bgrd, frame_bgrd, mask=bg_mask)
        gr_dep_mask = cv2.inRange(gr_dep_mask, 1000, 1200)
        gr_dep_mask = imf.open_img(gr_dep_mask, 15, 15)


        depth_edges = cv2.bitwise_and(frame_bgrd, frame_bgrd, mask=bg_mask)
        depth_edges = depth_edges - np.amin(depth_edges)
        depth_edges = depth_edges * 255.0 / (np.amax(depth_edges) - np.amin(depth_edges))
        depth_edges = np.uint8(depth_edges)
        canny = cv2.Canny(depth_edges, 20, 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        canny = cv2.dilate(canny, kernel, iterations=1)

        # Remove circular patterns
        imf.resize_image(canny, 'GR', 0.5)
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 0, minRadius=50)
        if circles is not None:
            for cir in circles[0]:
                cv2.circle(canny, (int(cir[0]), int(cir[1])), int(cir[2]), 150, 5)

        comb = cv2.bitwise_and(canny, gr_dep_mask)
        imf.resize_image(canny, 'GR_no circle', 0.5)

        #gr_dep_contours, _ = cv2.findContours(gr_dep_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #gr_draw = frame_bgr.copy()

        # Get mask for ROE, FS, and AF by HSI thresholding
        hsi_ub = np.array([hls_values[0], hls_values[2], hls_values[4]])
        hsi_lb = np.array([hls_values[1], hls_values[3], hls_values[5]])
        hsi_thresh = cv2.inRange(hsi_aoi, hsi_lb, hsi_ub)
        hsi_thresh = imf.open_img(hsi_thresh, 5, 5)  # denoise

        # sobelx = cv2.Sobel(frame_bgrd, cv2.CV_64F, 1, 0, ksize=7)
        # sobely = cv2.Sobel(frame_bgrd, cv2.CV_64F, 0, 1, ksize=7)
        # sobel = cv2.add(sobelx, sobely)
        # print(f"{np.min(sobel)} {np.max(sobel)}")
        # #sobel = cv2.dilate(sobel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
        # #smin = imf.retrieve_trackbar('min', 'trackbars')
        # #smax = imf.retrieve_trackbar('max', 'trackbars')
        # # TODO: MAYBE MAYBE MAYBE no
        # sobel_thresh_high = cv2.inRange(sobel, np.max(sobel)*0.50, np.max(sobel)*0.9)
        # sobel_thresh_low = cv2.inRange(sobel, -np.max(sobel)*0.9, -np.max(sobel)*0.50)
        # sobel_thresh = cv2.add(sobel_thresh_high, sobel_thresh_low)
        # sobel_thresh = cv2.dilate(sobel_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        # imf.resize_image(sobel_thresh, 'sobel', 0.5)

        # Combine masks
        combined_mask = cv2.bitwise_or(gr_dep_mask, hsi_thresh, mask=bg_mask)
        combined_mask = imf.close_img(combined_mask, 5, 5)
        comb_contours, comb_hierarchy = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        hsi_aoi = cv2.bitwise_and(frame_bgr, frame_bgr, mask=hsi_thresh)

        fg_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=bg_mask)
        diff = cv2.bitwise_and(frame_bgr, fg_bgr)
        #imf.resize_image(diff, 'results', 0.5)

        # Get edges in depth data
        # TODO Try this with hough to remove circles
        # depth_edges = cv2.bitwise_and(frame_bgrd, frame_bgrd, mask=bg_mask)
        # depth_edges = depth_edges - np.amin(depth_edges)
        # depth_edges = depth_edges * 255.0 / (np.amax(depth_edges) - np.amin(depth_edges))
        # depth_edges = np.uint8(depth_edges)
        # canny = cv2.Canny(depth_edges, 20, 255)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # canny = cv2.dilate(canny, kernel)

        kernel = imf.retrieve_trackbar('Kernel', 'trackbars', odd_only=True)
        bin = hsi_thresh#cv2.add(thresh_hsv, canny)
        bin = imf.open_img(bin, kernel, kernel)
        # contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contour_area = []
        # for i in range(len(contours)):
        #     if cv2.contourArea(contours[i]) >= 100:
        #         contour_area.append(contours[i])

        # cnt_list = combine_contours(contour_area, gr_dep_contours, frame_bgr.shape)
        if comb_hierarchy is not None:
            hierarchy = comb_hierarchy[0] #[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]
            draw_frame = frame_bgr.copy()
            for cnt, hrc in zip(comb_contours, hierarchy):
                if cv2.contourArea(cnt) >= 10:
                    mask = np.zeros(bin.shape, np.uint8)
                    cv2.drawContours(mask, [cnt], 0, 255, -1)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
                    rgbd_aoi = cv2.bitwise_and(frame_bgrd, frame_bgrd, mask=mask)

                    cv2.drawContours(draw_frame, cnt, -1, (0, 255, 0), 3)
                    # mask_display = imf.depth_to_display(rgbd_aoi)
                    # imf.resize_image(mask_display, 'mask', 0.5)
                    # cv2.waitKey(1)

                    features = FeatureSpace()
                    features.create_features(cnt, np.array(hrc[2] != -1), 'test')

                # detected, probability = classifier.classify(features.get_features())
                # if probability >= 0.5:
                #


        depth_display = imf.depth_to_display(frame_bgrd)

        #imf.resize_image(depth_display, 'aligned depth image', 0.5)
        #imf.resize_image(bin, 'thresh color image', 0.5)
        #imf.resize_image(draw_frame, 'results', 0.5)

        key = cv2.waitKey(1)
        if key == ord('p'):
            cv2.waitKey(5)
            key = cv2.waitKey(0)
        if key == ord('q'):
            exit(0)
        elif key == ord('s'):
            for i in range(10):
                frame_bgr = vid_bgr.read()
                frame_bgrd_8bit = vid_bgrd.read()
