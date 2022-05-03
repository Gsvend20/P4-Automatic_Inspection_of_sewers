import cv2
import numpy as np

# Paths (ugh)
path_co = 'C:/Users/simo1/OneDrive/Dokumenter/AAU Documents/Sem4 - Semester Project/P4-Automatic_Inspection_of_sewers/sewer recordings/Calibration/Color/colorboard'
path_ir = 'C:/Users/simo1/OneDrive/Dokumenter/AAU Documents/Sem4 - Semester Project/P4-Automatic_Inspection_of_sewers/sewer recordings/Calibration/IR/irboard'

img_co = []
img_ir = []
img_ir_scaled = []

for i in range(1, 5):
    img_co.append(cv2.imread(path_co + str(i) + '.png', -1))
    img_ir.append(cv2.imread(path_ir + str(i) + '.png', -1))

# cv2.namedWindow('chessboard pattern co', cv2.WINDOW_NORMAL)
# cv2.namedWindow('chessboard pattern ir', cv2.WINDOW_NORMAL)


def camera_calibration(images_co, images_ir):
    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 6  # number of checkerboard rows.
    columns = 9  # number of checkerboard columns.
    world_scaling = 27  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width_co = images_co[0].shape[1]
    height_co = images_co[0].shape[0]
    width_ir = images_ir[0].shape[1]
    height_ir = images_ir[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_co = []  # 2d points in image plane.
    imgpoints_ir = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints_co = []  # 3d point in real world space
    objpoints_ir = []  # 3d point in real world space

    for frame in images_co:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)
        if ret:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            #cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)
            #cv2.imshow('img', frame)
            #k = cv2.waitKey(0)

            objpoints_co.append(objp)
            imgpoints_co.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_co, imgpoints_co, (width_co, height_co), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)

    for frame in images_ir:
        # find the checkerboard
        cv2.imshow('test', frame)
        cv2.waitKey(0)
        ret, corners = cv2.findChessboardCorners(frame, (rows, columns), None)
        print(ret)
        if ret:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(frame, corners, conv_size, (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv2.imshow('img', frame)
            k = cv2.waitKey(0)

            objpoints_ir.append(objp)
            imgpoints_ir.append(corners)

    return mtx, dist


for i in range(0, len(img_ir)):
    # Turn into uint8 array
    im = cv2.convertScaleAbs(img_ir[i], alpha=0.05, beta=0)
    img_ir_scaled.append(im)

    # Show scaled image
    cv2.imshow('test', img_ir_scaled[i])
    cv2.waitKey(0)

    ret, corners = cv2.findChessboardCornersSB(img_ir_scaled[i], (6, 9), None)
    print(ret)
    if ret:
        # Convolution size used to improve corner detection. Don't make this too large.
        conv_size = (11, 11)

        # opencv can attempt to improve the checkerboard coordinates
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img_ir_scaled[i], corners, conv_size, (-1, -1), criteria)
        cv2.drawChessboardCorners(img_ir_scaled[i], (6, 9), corners, ret)
        cv2.imshow('img', img_ir_scaled[i])
        k = cv2.waitKey(0)



#camera_calibration(img_co, img_ir_scaled)
#mtx_co, dist_co = camera_calibration(img_co)
#mtx_ir, dist_ir = camera_calibration(img_ir_scaled)
