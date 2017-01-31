# Find corners of the chessboard images in order to correct distortion
# Calibrate camera & apply a distortion correction to raw images.
# Apply a perspective transform to rectify binary image (“birds-eye view”).
# Warp the detected lane boundaries back onto the original image.

import cv2
import numpy as np

LEFT_BOTTOM = (245, 700)
LEFT_TOP = (600, 447)
RIGHT_TOP = (683, 447)
RIGHT_BOTTOM = (1078, 700)

LEFT_TOP_WARPED = (245, 1)
RIGHT_TOP_WARPED = (1078, 1)

SRC = np.float32([LEFT_BOTTOM, LEFT_TOP, RIGHT_TOP, RIGHT_BOTTOM])
DST = np.float32([LEFT_BOTTOM, LEFT_TOP_WARPED, RIGHT_TOP_WARPED, RIGHT_BOTTOM])


def find_corners(img, nx, ny):
    """
                Function calculates (if possible) chessboard corners and returns them

    :param img: calibration image
    :param nx:  number of corners per row
    :param ny:  number of corners per column
    :return:    calculated corners if successful
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return cv2.findChessboardCorners(gray, (nx, ny), None)


def camera_calibration(img, objpoints, imgpoints):
    """
                      Returns calibrated camera matrix and distortion coefficients

    :param img:       image to get the shape
    :param objpoints: 3d points in real world space
    :param imgpoints: 2d points in image plane
    :return:          tuple of camera matrix and distortion coefficients
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist


def undistort_image(img, mtx, dist):
    """
                 Undistorts image by means of camera matrix and distortion coefficients

    :param img:  image to undistort
    :param mtx:  camera matrix
    :param dist: distortion coefficients
    :return:     undistorted image
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


def warp(img, invert=False):
    """
                Performs perspective transform of the image
    :param img:    image to apply perspective transform
    :param invert: check if warp should be inverted
    :return:       perspective transformed image
    """

    img_size = (img.shape[1], img.shape[0])
    if invert:
        M = cv2.getPerspectiveTransform(DST, SRC)
    else:
        M = cv2.getPerspectiveTransform(SRC, DST)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped
