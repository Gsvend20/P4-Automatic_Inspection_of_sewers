import math
import cv2
import numpy as np

def resize_image(image, image_name, procent):
    [height, width] = [image.shape[0],image.shape[1]]
    [height, width] = [procent*height, procent*width]
    cv2.namedWindow(image_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(image_name, int(width), int(height))
    cv2.imshow(image_name, image)

picNumb = (0,0)
def print_pic(winname, input_img):  # Function used under debugging for showing the different changes (Thres, Morp...)
    global picNumb
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(winname, 1 + picNumb[0] * (450 + 1), 1 + picNumb[1] * (450 + 10))
    picNumb[0] += 1
    if picNumb[0] >= math.floor(1920/450):
        picNumb[0] = 0
        picNumb[1] += 1

    returnImg = cv2.resize(input_img, (450, 450))
    cv2.imshow(winname, returnImg)
    return


def medColour(input_img, thresh_img):
    numberObj = 0
    medtotal = [0, 0, 0]
    for y, row in enumerate(input_img):
        for x, pixel in enumerate(row):
            if thresh_img[y, x] > 250:
                medtotal += input_img[y, x]
                numberObj += 1
    return [medtotal[0] / numberObj, medtotal[1] / numberObj, medtotal[2] / numberObj]


def threshold_values(input_img):
    low_values = (60, 50, 40)
    high_values = (150, 150, 170)
    return_img = cv2.inRange(input_img, low_values, high_values)
    return return_img


def sobel_values(input_img):
    edgeKernel = 101  # Kernel for Egdedetection
    sobelx = cv2.Sobel(input_img, cv2.CV_64F, 1, 0,edgeKernel)
    sobely = cv2.Sobel(input_img, cv2.CV_64F, 0, 1, edgeKernel)
    sobelMix = cv2.add(sobelx, sobely)
    return sobelMix


def canny_values(input_img):
    edgeLow = 100
    edgeHigh = 200
    cannyimg = cv2.Canny(input_img, edgeLow, edgeHigh)

    return cannyimg


def blur_values(input_img, blurType):
    kern = 7  # for Bluring the image
    if blurType == 'Gaussian':
        return cv2.GaussianBlur(input_img, (kern, kern), cv2.BORDER_DEFAULT)
    if blurType == 'Median':
        return cv2.medianBlur(input_img, kern, cv2.BORDER_DEFAULT)
    if blurType == 'Blur':
        return cv2.blur(input_img, (kern, kern), cv2.BORDER_DEFAULT)


def open_img(input_img, kernelErode, kernelDila=None):
    if kernelDila is None:
        kernelDila = kernelErode
    # out_img = np.zeros(input_img)
    kernelErode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelErode, kernelErode))
    kernelDila = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelDila, kernelDila))
    Erodeimg = cv2.erode(input_img, kernelErode)
    return cv2.dilate(Erodeimg, kernelDila)


def close_img(input_img, kernelErode, kernelDila=None):
    if kernelDila is None:
        kernelDila = kernelErode
    # out_img = np.zeros(input_img)
    kernelErode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelErode, kernelErode))
    kernelDila = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernelDila, kernelDila))
    Dilaimg = cv2.dilate(input_img, kernelDila)
    return cv2.erode(Dilaimg, kernelErode)


# A function for reversing the 16bit to 8bit changes
def convert_to_16(img):
    if img.shape[2] == 3:
      depth_hi_bytes, depth_lo_bytes, empty = cv2.split(img)
      return depth_lo_bytes.astype('uint16') + np.left_shift(depth_hi_bytes.astype('uint16'), 8)
    else:
      print('Image ' + ' is not 3 channel')
      return img


# This is just for trackbar callbacks
def _nothing(x):
    pass


def retrieve_trackbar(trackbar_name,img_window_name, odd_only=False):
    track_num = cv2.getTrackbarPos(trackbar_name, img_window_name)
    if not odd_only:
        return track_num
    else:
        if track_num == 0:
            return track_num
        else:
            count = track_num % 2
            if count == 0:
                track_num += 1
                return track_num
            else:
                return track_num


def define_trackbar(trackbar_name, img_window_name, max_min_values):
    cv2.namedWindow(img_window_name, cv2.WINDOW_AUTOSIZE)
    #cv2.resizeWindow(img_window_name, 400, 400)
    min_val, max_val = max_min_values
    cv2.createTrackbar(trackbar_name, img_window_name, min_val, max_val, _nothing)


def depth_to_display(depth_frame):
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

    # Aligned and depth images have different shapes, so we check for both
    shape = depth_frame_b.shape
    if len(shape) <= 1:
        depth_frame_color = cv2.merge([depth_frame_b[:, :, 0], depth_frame_g[:, :, 0], depth_frame_r[:, :, 0]])
    else:
        depth_frame_color = cv2.merge([depth_frame_b[:, :], depth_frame_g[:, :], depth_frame_r[:, :]])

    depth_frame_color = depth_frame_color.astype(np.uint8)
    return depth_frame_color

def save_depth_img(path, image):
    rgbd_hi_bytes = np.right_shift(image, 8).astype('uint8')
    rgbd_lo_bytes = image.astype('uint8')

    shape = image.shape
    if len(shape) <= 1:
        split_rgbd_image = cv2.merge([rgbd_hi_bytes[:, :, 0], rgbd_lo_bytes[:, :, 0], np.zeros_like(rgbd_hi_bytes[:, :, 0])])
    else:
        split_rgbd_image = cv2.merge([rgbd_hi_bytes[:, :], rgbd_lo_bytes[:, :], np.zeros_like(rgbd_hi_bytes[:, :])])
    cv2.imwrite(path,split_rgbd_image)


class AdaptiveGRDepthMasker:
    def __init__(self, max_images, start_range, area_range):
        self._mask_list = [np.zeros((1920, 1080), dtype='uint8')]
        self._masked_mean_depth_list = [0]
        self._list_length = max_images
        self._range = start_range
        self._area_min = area_range[0]
        self._area_max = area_range[1]

    def add_image(self, image):  # Call function each frame to update mask
        # Generate mask
        mask = cv2.inRange(image, self._range[0], self._range[1])
        inv_mask = cv2.bitwise_not(mask)
        masked_image = np.ma.array(image, dtype='uint16', mask=inv_mask)

        # Check area by calculating number of masked pixels
        mask_area = 1920 * 1080 - np.ma.count_masked(masked_image)
        if self._area_min <= mask_area < self._area_max:

            # Save mask for prediction
            self._mask_list.append(mask)

            # Save mean depth in masked area
            self._masked_mean_depth_list.append(np.ma.mean(masked_image))

            # Remove oldest mask to keep array at desired length
            if len(self._mask_list) > self._list_length:
                self._mask_list.pop(0)
                self._masked_mean_depth_list.pop(0)

                # Calculate new range for area and depth
                dist_array = np.zeros(self._list_length - 1)
                for i in range(self._list_length - 1):
                    # Find differences between the mean distances of the saved images
                    dist_array[i] = self._masked_mean_depth_list[i + 1] - self._masked_mean_depth_list[i]

                mean_dist_change = np.mean(dist_array)

                # Adjust searching range and area
                if not np.isnan(mean_dist_change):
                    self._range = (int(self._range[0] + mean_dist_change), int(self._range[1] + mean_dist_change))

    def return_masks(self):
        mask = np.zeros_like(self._mask_list[-1])
        contours, _ = cv2.findContours(self._mask_list[-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            # If the back-wall starts coming through, maybe look into reducing the size checked
            #print(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            c_x, c_y = int(x+w/2), int(y+h/2)
            if 1080/2 + 100 <= c_x or c_x <= 1080/2 - 100 or 1920/2 + 150 <= c_y or c_y <= 1920/2 - 150:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        return mask


def find_largest_contour(contours, hierarchy):
    largest_a = 0
    for n in range(len(hierarchy)):
        a = cv2.contourArea(contours[n])
        if largest_a < a:
            largest_a = a
            largest_no = n
    return contours[largest_no], hierarchy[largest_no]


def average_contour_depth(depth_img, cnt):
    # Get the average depth
    d_mask = np.zeros((1920,1080), dtype='uint8')
    cv2.fillPoly(d_mask, cnt, 255)
    depth_cnt = cv2.bitwise_and(depth_img, depth_img, mask=d_mask)
    if np.max(depth_cnt) != 0:
        mean = np.mean(np.nonzero(depth_cnt))
    else:
        mean = 10  # Set to close
    return mean
