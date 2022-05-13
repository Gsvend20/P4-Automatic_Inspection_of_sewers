from Functions.Featurespace import Classifier
from Functions.Featurespace import FeatureSpace
from Functions import imgproc_func as imf
import cv2
import numpy as np
import os


# Path to video folders
folders_path = 'Test'

# Load saved video streams
vid_bgr = cv2.VideoCapture()
vid_bgrd = cv2.VideoCapture()

#trackbars = ['high_H', 'low_H','high_L', 'low_L', 'high_S', 'low_S']
trackbars = ['sigma']
hls_values = [255, 70, 255, 37, 255, 24]

# Trackbars for adjusting values
imf.define_trackbar('Kernel', 'trackbars', (3, 31))
for tracks in trackbars:
    imf.define_trackbar(tracks, 'trackbars', (150, 1000))

P=0
# load classifier
classifier = Classifier()
classifier.load_trained_classifier('classifier.pkl')
print(classifier._classifier)

while True:
    vid_bgr.open(f"{folders_path}/24_1_bgr.avi")
    vid_bgrd.open(f"{folders_path}/24_1_aligned.avi")

    while vid_bgr.isOpened():
        # Fetch next frame
        if not P:
            ret, frame_bgr = vid_bgr.read()
            if not ret:
                break
            ret, frame_bgrd_8bit = vid_bgrd.read()
            if not ret:
                break
        # Convert depthmap to 16 bit depth
        frame_bgrd = imf.convert_to_16(frame_bgrd_8bit)

        #frame_bgrd = cv2.GaussianBlur(frame_bgrd, (5, 5), 0)
        blur = cv2.medianBlur(frame_bgr, 7)

        frame_hsi = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        frame_grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        trackbar_data = [None] * len(trackbars)
        for i, tracks in enumerate(trackbars):
            trackbar_data[i] = imf.retrieve_trackbar(tracks, 'trackbars')

        # Generate area of interest from pipe depth data
        aoi_end = cv2.inRange(frame_bgrd, int(np.max(frame_bgrd) - 100), int(np.max(frame_bgrd)))
        aoi_pipe = cv2.inRange(frame_bgrd, 600, int(np.max(frame_bgrd) - 100))
        cnt, hir = cv2.findContours(aoi_pipe, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pipe_mask = np.zeros_like(frame_bgrd).astype('uint8')
        pipe_mask = cv2.fillPoly(pipe_mask, cnt, 255)
        bg_mask = cv2.subtract(pipe_mask, aoi_end)
        bg_mask = cv2.dilate(bg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        hsi_aoi = cv2.bitwise_and(frame_hsi, frame_hsi, mask=bg_mask)

        #Vlow = imf.retrieve_trackbar('low', 'trackbars')
        upperLimit1 = np.array([hls_values[0], hls_values[2], hls_values[4]])
        lowerLimit1 = np.array([hls_values[1], hls_values[3], hls_values[5]])  # Threshold around highlights

        hsi_thresh = cv2.inRange(hsi_aoi, lowerLimit1, upperLimit1)

        # Get edges in depth data
        v = np.median(frame_hsi[:, :, 1])
        sigma = trackbar_data[0]
        low = int(max(0, (1.0 - sigma/100) * v))
        high = int(min(255, (1.0 + sigma/100) * v))

        depth_edges = frame_bgrd - np.amin(frame_bgrd)
        depth_edges = depth_edges * 255.0 / (np.amax(depth_edges) - np.amin(depth_edges))
        depth_edges = np.uint8(depth_edges)

        edgeKernel = 3  # Kernel for Egdedetection
        sobelx = cv2.Sobel(depth_edges, -1, 0, 1, edgeKernel)
        sobely = cv2.Sobel(depth_edges, -1, 1, 0, edgeKernel)
        sobel = cv2.addWeighted(sobelx, 1, sobely, 1, 0)
        scharr = cv2.Laplacian(depth_edges, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        canny = cv2.Canny(frame_hsi[:,:,1], low, high)

        kernel = imf.retrieve_trackbar('Kernel', 'trackbars', odd_only=True)
        bin = cv2.add(hsi_thresh, canny)
        bin = imf.open_img(bin, kernel, kernel)
        contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if hierarchy is not None:
            hierarchy = hierarchy[0]  #[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]
            draw_frame = frame_bgr.copy()
            for cnt, hrc in zip(contours, hierarchy):
                if cv2.contourArea(cnt) >= 100:
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

                    detected, probability = classifier.classify(np.asarray(features.get_features()[0]))
                    if probability > 0.25:
                        print(features.get_features())
                        print(detected, probability)

            imf.resize_image(draw_frame, 'results', 0.5)


        depth_display = imf.depth_to_display(frame_bgrd)
        imf.resize_image(depth_display, 'aligned depth image', 0.5)
        imf.resize_image(bin, 'thresh color image', 0.5)
        imf.resize_image(canny, 'edges',0.5)

        key = cv2.waitKey(1)
        if key == ord('p'):
            if P == 1:
                P = 0
            else:
                P = 1
        if key == ord('q'):
            exit(0)
        elif key == ord('s'):
            for i in range(10):
                frame_bgr = vid_bgr.read()
                frame_bgrd_8bit = vid_bgrd.read()

