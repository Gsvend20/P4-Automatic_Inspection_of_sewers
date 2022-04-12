import cv2
import numpy as np


def initialize_streams(path_to_bgr, path_to_depth, path_to_ir):
    bgr_video = cv2.VideoCapture(path_to_bgr)
    depth_video = cv2.VideoCapture(path_to_depth)
    ir_video = cv2.VideoCapture(path_to_ir)
    return bgr_video, depth_video, ir_video


def extract_next_frame(video):
    retval, frame = video.read()
    if retval:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # rotate frame to upright position
        return frame
    else:
        print('No frames to read!')
        exit(1)
        return


def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0], image.shape[1]]
    [height, width] = [procent * height, procent * width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)
    return


def open_image(input_image, e_kernel, d_kernel):
    e_out = cv2.erode(input_image, e_kernel)
    d_out = cv2.dilate(e_out, d_kernel)
    return d_out


def close_image(input_image, e_kernel, d_kernel):
    d_out = cv2.dilate(input_image, d_kernel)
    e_out = cv2.erode(d_out, e_kernel)
    return e_out


def find_circularity(area, perimeter):
    circ = 4*np.pi*area/pow(perimeter, 2)
    return circ


def find_compactness(area, width, height):
    comp = area/(width*height)
    return comp


def find_elongation(cnt):
    (x, y), (width, height), angle = cv2.minAreaRect(cnt)
    elongation = min(width, height) / max(width, height)
    return elongation


def find_ferets(cnt):
    (x, y), (width, height), angle = cv2.minAreaRect(cnt)
    if width >= height:
        return width
    else:
        return height


def get_features(contour):
    feature_matrix = np.array(range(0, 7), dtype='float')

    # Calculate feature properties
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    (min_rect_x, min_rect_y), (min_rect_width, min_rect_height), min_rect_angle = cv2.minAreaRect(contour)
    (ellipse_x, ellipse_y), (ellipse_MA, ellipse_ma), ellipse_angle = cv2.fitEllipse(contour)
    (circle_x, circle_y), circle_radius = cv2.minEnclosingCircle(cnt)

    feature_matrix[0] = area
    feature_matrix[1] = perimeter
    feature_matrix[2] = find_circularity(area, perimeter)
    feature_matrix[3] = find_compactness(area, min_rect_width, min_rect_height)
    feature_matrix[4] = find_elongation(contour)
    feature_matrix[5] = find_ferets(contour)
    feature_matrix[6] = ellipse_ma / ellipse_MA
    return feature_matrix


def detect_GR_from_depth(contours, input_image):  # Function to detect depth discrepancies away from the center of the pipe
    draw_circle_mat = np.zeros_like(input_image)  # Create blank image to draw on
    draw_ellipse_mat = np.zeros_like(input_image)
    height, width = input_image.shape[:]  # Detect image size

    for cnt in contours:
        contour_area = cv2.contourArea(cnt)  # Calculate contour area

        if contour_area >= 1000:  # Only focus on contours containing 1000 or more pixels
            # Enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            circle_center = np.array([int(x), int(y)])
            lower_bound = np.array([int(width / 2 - 50), int(3 * height / 5 - 50)])
            upper_bound = np.array([int(width / 2 + 50), int(3 * height / 5 + 50)])

            # Best fit ellipse
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            cv2.ellipse(draw_ellipse_mat, (int(x), int(y)), (int(MA), int(ma)), int(angle), 0, 360, color=(255, 0, 0),
                        thickness=2)

            # Check if contour is in the center of the pipe
            #if (circle_center <= lower_bound).any() or (circle_center >= upper_bound).any():
            #    # print('x-coordinate: ' + str(circle_center[0]) + ' y-coordinate: ' + str(circle_center[1]))
            #    cv2.circle(draw_circle_mat, circle_center, int(radius), (255, 0, 0), 2)
            #    cv2.ellipse(draw_ellipse_mat, (int(x), int(y)), (int(MA), int(ma)), int(angle), 0, 360, color=(255, 0, 0), thickness=2)
            #else:
            #    circle_center = np.array([0, 0])
    return draw_ellipse_mat


# Paths to the video files
path_bgr = 'video_dump/bgr_GR_upstream_left.4.8.9.23.avi'
path_depth = 'video_dump/depth_GR_upstream_left.4.8.9.23.avi'
path_ir = 'video_dump/ir_GR_upstream_left.4.8.9.23.avi'

# Import video streams
vid_bgr, vid_depth, vid_ir = initialize_streams(path_bgr, path_depth, path_ir)

# Kernel sizes and shapes
median_blur_kernel_size = 19  # Kernel for noise removal
open_close_structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

while True:
    # read frames
    bgr_frame = extract_next_frame(vid_bgr)
    depth_frame = extract_next_frame(vid_depth)
    ir_frame = extract_next_frame(vid_ir)

    # make grayscale
    bgr_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
    ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)

    # Median blur to remove noise
    bgr_frame = cv2.medianBlur(bgr_frame, median_blur_kernel_size)
    depth_frame = cv2.medianBlur(depth_frame, median_blur_kernel_size)
    ir_frame = cv2.medianBlur(ir_frame, median_blur_kernel_size)

    # Adaptive thresholding
    bgr_frame = cv2.adaptiveThreshold(bgr_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
    #depth_frame = cv2.adaptiveThreshold(depth_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
    ir_frame = cv2.adaptiveThreshold(ir_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

    depth_frame_thresh = cv2.inRange(depth_frame, 25, 60)

    # Closing
    bgr_frame = close_image(bgr_frame, open_close_structuring_element, open_close_structuring_element)
    depth_frame_thresh = close_image(depth_frame_thresh, open_close_structuring_element, open_close_structuring_element)
    ir_frame = close_image(ir_frame, open_close_structuring_element, open_close_structuring_element)

    # Detect contours
    depth_contours, hierarchy = cv2.findContours(depth_frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in depth_contours:
        if cv2.contourArea(cnt) >= 1000:
            depth_features = get_features(cnt)
            print(depth_features)

    GR_ellipse_mat = detect_GR_from_depth(depth_contours, depth_frame)

    cv2.imshow('detected contours', cv2.add(depth_frame, GR_ellipse_mat))

    # Show video feeds
    #resize_image(bgr_frame, 'bgr', 0.5)
    #resize_image(depth_frame, 'depth', 1)
    #resize_image(ir_frame, 'ir', 1)

    pressed_key = cv2.waitKey(1)
    if pressed_key == ord('w'):
        for i in range(50):
            vid_bgr.read()
            vid_depth.read()
            vid_ir.read()
    if pressed_key == ord('q'):
        break
