from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
from Functions import imgproc_func as imf
import cv2
import numpy as np
import os
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter


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


class TemporalMean:
    def __init__(self, samples, image_shape):
        self.mean_image = np.zeros(image_shape, dtype='uint16')
        self.mean_image_8bit = np.zeros(image_shape, dtype='uint8')
        self.difference = np.zeros(image_shape, dtype='uint16')
        self.difference_8bit = np.zeros(image_shape, dtype='uint8')
        self._image_list = []
        self._list_length = samples

    def add_image(self, image):
        if len(self._image_list) < self._list_length:
            self._image_list.append(image)
        else:
            self._image_list.pop(0)
            self._image_list.append(image)
        self.mean_image = np.mean(self._image_list, axis=0, dtype='uint16')
        difference = cv2.subtract(image, self.mean_image)
        self.difference = difference.astype('uint16')
        mean_image_8bit = self.mean_image * 255.0 / (np.amax(self.mean_image) - np.amin(self.mean_image))
        self.mean_image_8bit = mean_image_8bit.astype('uint8')
        difference_8bit = self.difference * 255.0 / (np.amax(self.difference) - np.amin(self.difference))
        self.difference_8bit = difference_8bit.astype('uint8')


# Add image
# Check within specified range band
# Wait for area of a certain size to appear
    # Not directly overhead or below (avoid AF and ROE) (mask it out or check CoM)
# Once seen keep track of area, mean depth and depth variance in region
# Look at following frames to predict depth change of area per frame
# Adjust frame range to keep area within frame
    # Find a way to stop this when the area gets too scuffed (look at size or touching edge of image maybe?)
class AdaptiveGRDepthMasker:
    def __init__(self, max_images, start_range, min_start_area):
        self._image_list = []
        self._mask_list = []
        self.masked_image_list = []
        self._list_length = max_images
        self._range = start_range
        self.mask = np.zeros((1920, 1080))
        self._area_min = min_start_area
        self._area_max = 15000

    def add_image(self, image):  # Call function each frame to update mask
        # Generate mask
        mask = cv2.inRange(image, self._range[0], self._range[1])
        #masked_image = cv2.bitwise_and(image, image, mask=mask)
        inv_mask = cv2.bitwise_not(mask)
        masked_image = np.ma.array(image, dtype='uint16', mask=inv_mask)

        # Save masks for prediction
        self._image_list.append(image)
        self._mask_list.append(mask)
        self.masked_image_list.append(masked_image)

        # Remove oldest mask to keep array at desired length
        if len(self._image_list) > self._list_length:
            self._image_list.pop(0)
            self._mask_list.pop(0)
            self.masked_image_list.pop(0)

            # Check area by calculating number of masked pixels
            mask_area = 1920 * 1080 - np.ma.count_masked(masked_image)
            if self._area_min <= mask_area < self._area_max:

                # Calculate new range for area and depth
                dist_array = np.zeros(self._list_length - 1)
                area_array = np.zeros(self._list_length - 1)
                for i in range(self._list_length - 1):
                    # Find differences between the mean distances of the saved images
                    dist_array[i] = np.ma.mean(self.masked_image_list[i + 1]) - np.ma.mean(self.masked_image_list[i])

                    # Find differences between areas
                    #area_array[i] = 1920 * 1080 - np.ma.count_masked(masked_image)
                mean_dist_change = np.mean(dist_array)
                #last_mean_dist = np.ma.mean(self.masked_image_list[-1])
                #median_area_change = np.mean(area_array)
                #print(median_area_change)

                # Adjust searching range and area
                if not np.isnan(mean_dist_change):
                    self._range = (int(self._range[0] + mean_dist_change), int(self._range[1] + mean_dist_change))
                    #self._range = (int(last_mean_dist + mean_dist_change - 50), int(last_mean_dist + mean_dist_change + 50))
                    #self._area_min =
                    #self._area_max =

    def show(self):
        # Show image
        imf.resize_image(self._image_list[-1], 'GR_image', 0.5)

        # Show mask
        imf.resize_image(self._mask_list[-1], 'GR_mask', 0.5)

        # Show depth info in mask
        #print(np.ma.mean(self.masked_image_list[-1]))


depth_masker = AdaptiveGRDepthMasker(3, (1400, 1500), 3200)
depth_mean = TemporalMean(3, (1920, 1080))

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
imf.define_trackbar('H min', 'trackbars', (0, 255))
imf.define_trackbar('H max', 'trackbars', (1, 255))
imf.define_trackbar('I min', 'trackbars', (0, 255))
imf.define_trackbar('I max', 'trackbars', (1, 255))
imf.define_trackbar('S min', 'trackbars', (0, 255))
imf.define_trackbar('S max', 'trackbars', (1, 255))


hls_values = [255, 70, 255, 38, 255, 24]

# load classifier
#classifier = Classifier()
#classifier.load_trained_classifier('training_data.pkl')

while True:
    vid_bgr.open(f"{folders_path}/2_bgr.avi")
    vid_bgrd.open(f"{folders_path}/2_aligned.avi")

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
        aoi_end = cv2.inRange(frame_bgrd, int(np.max(frame_bgrd) - 320), int(np.max(frame_bgrd) + 100))
        aoi_pipe = cv2.inRange(frame_bgrd, 600, int(np.max(frame_bgrd) - 300))
        pipe_contours, _ = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pipe_mask = np.zeros_like(frame_bgrd).astype('uint8')
        pipe_mask = cv2.fillPoly(pipe_mask, pipe_contours, 255)
        bg_mask = cv2.subtract(pipe_mask, aoi_end)
        #bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        hsi_aoi = cv2.bitwise_and(frame_hsi, frame_hsi, mask=bg_mask)

        # Get mask for ROE, FS, and AF by HSI thresholding
        hsi_ub = np.array([hls_values[0], hls_values[2], hls_values[4]])
        hsi_lb = np.array([hls_values[1], hls_values[3], hls_values[5]])
        hsi_thresh = cv2.inRange(hsi_aoi, hsi_lb, hsi_ub)
        hsi_thresh = imf.open_img(hsi_thresh, 5, 5)  # denoise

        # Create BLOB for GR from depth frames
        gr_dep_mask = cv2.bitwise_and(frame_bgrd, frame_bgrd, mask=bg_mask)
        gr_dep_mask = cv2.inRange(gr_dep_mask, 1000, 1500)
        #imf.resize_image(gr_dep_mask, 'GR_detection_band', 0.5)
        # gr_dep_mask = imf.open_img(gr_dep_mask, 15, 15)
        # H = (imf.retrieve_trackbar('H min', 'trackbars'), imf.retrieve_trackbar('H max', 'trackbars'))
        # I = (imf.retrieve_trackbar('I min', 'trackbars'), imf.retrieve_trackbar('I max', 'trackbars'))
        # S = (imf.retrieve_trackbar('S min', 'trackbars'), imf.retrieve_trackbar('S max', 'trackbars'))
        # gr_mask = cv2.inRange(cv2.bitwise_and(frame_hsi, frame_hsi, mask=bg_mask), (H[0], I[0], S[0]), (H[1], I[1], S[1]))

        # Canny Edges on depth image
        depth_edges = cv2.bitwise_and(frame_bgrd, frame_bgrd, mask=bg_mask)
        depth_edges = depth_edges - np.amin(depth_edges)
        depth_edges = depth_edges * 255.0 / (np.amax(depth_edges) - np.amin(depth_edges))
        depth_edges = np.uint8(depth_edges)
        depth_edges = cv2.Canny(depth_edges, 20, 255, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        depth_edges = imf.close_img(depth_edges, 21, 21)
        depth_edges = cv2.dilate(depth_edges, kernel, iterations=3)
        # imf.resize_image(depth_edges, 'Depth_edges', 0.5)

        # Canny edges on intensity image
        intensity_edges = cv2.Canny(hsi_aoi[:,:,1], 20, 255, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        intensity_edges = imf.close_img(intensity_edges, 11, 11)
        intensity_edges = cv2.dilate(intensity_edges, kernel, iterations=3)
        # imf.resize_image(intensity_edges, 'Pipe4', 0.5)

        fg_d_frame = cv2.bitwise_and(frame_bgrd, frame_bgrd, mask=bg_mask)
        depth_masker.add_image(fg_d_frame)
        depth_masker.show()

        # grey_edges = cv2.bitwise_and(grey_frame, grey_frame, mask=bg_mask)
        # grey_edges = cv2.Canny(grey_edges, 20, 255, apertureSize=3)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # grey_edges = cv2.dilate(grey_edges, kernel, iterations=3)

        gr_edge = cv2.add(intensity_edges,depth_edges)

        # Remove circular patterns
        # circles = cv2.HoughCircles(gr_edge, cv2.HOUGH_GRADIENT, 1, 1, minRadius=10, maxRadius=300)
        # draw_circles = np.zeros_like(bg_mask)
        # if circles is not None:
        #     for cir in circles[0]:
        #         cv2.circle(draw_circles, (int(cir[0]), int(cir[1])), int(cir[2]), 150, 10)
        # imf.resize_image(draw_circles, 'circles', 0.5)

        # result = hough_ellipse(gr_edge, 250, 22, min_size=10)
        # result.sort(order='accumulator')
        # best = list(result[-1])
        # yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        # orientation = best[5]
        # cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        # ellipse = np.zeros_like(bg_mask)
        # ellipse[cy, cx] = 255
        # imf.resize_image(ellipse, 'ellipse', 0.5)

        gr_mask = cv2.subtract(gr_dep_mask, gr_edge)
        #imf.resize_image(gr_mask, 'GR_edgeremoved', 0.5)

        # find center of mass for pipe
        for cnt in pipe_contours:
            area = cv2.contourArea(cnt)
            if area > 0:
                pipe_x, pipe_y, pipe_len_x, pipe_len_y = cv2.boundingRect(cnt)
                pipe_com = (int(pipe_x + pipe_len_x / 2), int(pipe_y + pipe_len_y / 2))

        gr_contours, _ = cv2.findContours(gr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in gr_contours:
            area = cv2.contourArea(cnt)
            if area > 0:
                x, y, len_x, len_y = cv2.boundingRect(cnt)
                center_of_mass = (int(x + len_x / 2), int(y + len_y / 2))

                if not pipe_com[0] - 100 < center_of_mass[0] <= pipe_com[0] + 100:
                    if not pipe_com[1] - 100 < center_of_mass[1] <= pipe_com[1] + 100:
                        cv2.drawContours(hsi_thresh, [cnt], -1, 255, -1)

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

        #imf.resize_image(hsi_thresh, 'Pipe2', 0.5)

        # Combine masks
        combined_mask = cv2.bitwise_or(gr_dep_mask, hsi_thresh, mask=bg_mask)
        combined_mask = imf.close_img(combined_mask, 5, 5)
        comb_contours, comb_hierarchy = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        hsi_aoi = cv2.bitwise_and(frame_bgr, frame_bgr, mask=hsi_thresh)

        fg_bgr = cv2.bitwise_and(frame_bgr, frame_bgr, mask=bg_mask)
        imf.resize_image(fg_bgr, 'Pipe', 0.5)
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
