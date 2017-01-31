# Find corners of the chessboard images in order to correct distortion
# Calibrate camera & apply a distortion correction to raw images.
# Apply a perspective transform to rectify binary image (“birds-eye view”).
# Warp the detected lane boundaries back onto the original image.

import cv2


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


def warper(img, src, dst):
    """
                Performs perspective transform of the image
    :param img: image to apply perspective transform
    :param src: source image point (4 points)
    :param dst: destination image points (4 points) for mapping source points
    :return:    perspective transformed image
    """

    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)

    return warped
