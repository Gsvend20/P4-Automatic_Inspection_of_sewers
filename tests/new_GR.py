import cv2
from Functions import imgproc_func as imf
from math import atan2, degrees, pi
import numpy as np

def angle_frame(center, size):
    pixel_angle = np.zeros(size)

    for y, row in enumerate(pixel_angle):
        for x, pixel in enumerate(row):
            pixel_angle[y, x] = atan2(y-center[0], x-center[1])
            if pixel_angle[y, x] < 0: pixel_angle[y, x] += pi * 2
            pixel_angle[y, x] += pi
            pixel_angle[y, x] = degrees(pixel_angle[y, x])
            if pixel_angle[y, x] >= 360: pixel_angle[y, x] -= 360
            #pixel_angle[y, x] = pixel_angle[y, x] / 2.  # make it fit opencv
    return pixel_angle

def custom_sobel(shape, axis):
    """
    shape must be odd: eg. (5,5)
    axis is the direction, with 0 to positive x and 1 to positive y
    """
    k = np.zeros(shape)
    p = [(j,i) for j in range(shape[0])
           for i in range(shape[1])
           if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]

    for j, i in p:
        j_ = int(j - (shape[0] -1)/2.)
        i_ = int(i - (shape[1] -1)/2.)
        k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)
    return k


vid_path = r'C:\Users\Muku\OneDrive - Aalborg Universitet\P4 - GrisProjekt\Exam videos\GR_3_aligned (1).avi'

depth_src = cv2.VideoCapture(vid_path)
img_size = (1920, 1080)

ret, frame_depth = depth_src.read()
frame_depth = imf.convert_to_16(frame_depth)  # Convert the depth data back into readable data

# Finding the center of the pipe with a hough transform
drawframe = imf.depth_to_display(frame_depth)
d_bin = cv2.threshold(frame_depth, 20, 255, cv2.THRESH_BINARY)[1].astype('uint8')

# I need retr_tree to fill out empty pixels
contours, hrc = cv2.findContours(d_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
hrc = hrc[0]
parent_bool = np.array(hrc[:][:, 3] == 1)

best_fit = 0
for n, state in enumerate(parent_bool):
    cnt = contours[n]
    # Checks to see if I need to run the contour
    if state:
        continue
    area = cv2.contourArea(cnt)
    # Since the diameter is usually around 500 pixels I can easily ignore anything that has an area below that
    if area < 600:
        continue
    # since we are looking for a circle I am checking this aswell
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area / (perimeter ** 2))
    print(circularity)
    if circularity < 0.6:
        continue

    # create a frame to keep the holes
    fill_frame = imf.depth_to_display(frame_depth)
    contour_holes = np.array(contours)[np.array(hrc[:][:, 3] == n)]
    # Draw in the contour and holes
    cv2.drawContours(fill_frame, [cnt], 0, 1,-1)
    cv2.drawContours(fill_frame, contour_holes, -1, 0, -1)

    fillness = np.sum(fill_frame)/area
    if fillness > best_fit:
        best_fit = fillness
        contour = cnt

#cv2.drawContours(drawframe, [contour], 0, 255,-1)
((x, y), radius) = cv2.minEnclosingCircle(contour)
center = (int(x),int(y))
cv2.circle(drawframe, (center[0],center[1]), 0, (255,0,0), 3)


# Old center calculations
#circles = cv2.HoughCircles(frame_depth.astype('uint8'),cv2.HOUGH_GRADIENT,1, 1600, param1=100,param2=100,minRadius=0,maxRadius=500)
#circles = np.uint16(np.around(circles))
#center = circles[0][0]
#cv2.circle(drawframe, (center[0],center[1]), center[2], (255,0,0), 3)
#imf.resize_image(drawframe, 'ahhh', 0.4)
#cv2.waitKey(0)

# return a frame with every pixel telling their orientation to the center
pixel_angle = angle_frame((center[1],center[0]), img_size)

while True:
    ret, frame_depth = depth_src.read()
    if not ret:
        break
    frame_depth = imf.convert_to_16(frame_depth)  # Convert the depth data back into readable data
    frame_depth = cv2.medianBlur(frame_depth, 5)

    # Masking the angled pixels to pixels with depth data
    ret, mask = cv2.threshold(frame_depth, 20, 255, cv2.THRESH_BINARY)
    angle_copy = pixel_angle.copy()
    angle_copy = cv2.bitwise_and(angle_copy, angle_copy, mask=mask.astype('uint8'))

    # Defining kernels
    sobel_x = custom_sobel((9, 9), 0)
    sobel_y = custom_sobel((9, 9), 1)

    # Filter the blurred grayscale images using filter2D
    filtered_blurred_x = cv2.filter2D(frame_depth, cv2.CV_32F, sobel_x)
    filtered_blurred_y = cv2.filter2D(frame_depth, cv2.CV_32F, sobel_y)

    mag = cv2.magnitude(filtered_blurred_x, filtered_blurred_y)
    orien = cv2.phase(filtered_blurred_x, filtered_blurred_y, angleInDegrees=True)
    #orien = orien / 2. # Go from 0:360 to 8 bit
    #orien = cv2.medianBlur(orien, 5) # there's a bunch of noise in this

    angled = orien - angle_copy
    #for y, row in enumerate(angled):
    #    for x, pixel in enumerate(row):
    #        if angled[y, x] < 0: angled[y, x] += 360

    hsv = np.zeros((1920,1080,3), dtype='uint8')
    hsv[..., 0] = angled / 2. # H (in OpenCV between 0:180)
    hsv[..., 1] = 255  # S
    hsv[..., 2] = mask.astype('uint8')  # V 0:255
    treshold = cv2.inRange(angled, 0, 40)
    treshold2 = cv2.inRange(angled, 360-40, 0)
    treshold = cv2.add(treshold, treshold2)
    treshold = cv2.subtract(mask.astype('uint8'), treshold)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    imf.resize_image(bgr, 'both', 0.4)
    imf.resize_image(treshold, 'thresh', 0.4)

    angled = orien - angle_copy
    hsv[..., 0] = orien / 2.  # H (in OpenCV between 0:180)
    hsv[..., 2] = mask.astype('uint8')
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    imf.resize_image(bgr, 'orien', 0.4)
    hsv[..., 0] = pixel_angle / 2.  # H (in OpenCV between 0:180)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    imf.resize_image(bgr, 'fixed', 0.4)
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite('original.png', orien)
        cv2.imwrite('orientation.png', bgr)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(25)
        cv2.waitKey(0)