import numpy as np
import cv2
import glob
from image_correction import find_corners
from image_correction import camera_calibration
from image_correction import undistort_image
from image_correction import warp
from transforamtion_pipeline import transform_image

from utils import save_image, draw_lines

# SAVE_FILE_CALIBRATION = 'output_images/1_undistorting.png'
# SAVE_FILE_TEST = 'output_images/2.1_undistorting_test.png'
# SAVE_FILE_TEST2 = 'output_images/2.2_undistorting_test2.png'
# SAVE_FILE_TEST_TRANSFORMED = 'output_images/3.1_transformed_test.png'
# SAVE_FILE_TEST2_TRANSFORMED = 'output_images/3.2_transformed_test2.png'
SAVE_PERSPECTIVE_TRANSFORM_1 = 'output_images/4.1_transformed_straight_lines1.png'
SAVE_PERSPECTIVE_TRANSFORM_2 = 'output_images/4.2_transformed_straight_lines2.png'

SAVE_FILE_CALIBRATION = False
SAVE_FILE_TEST = False
SAVE_FILE_TEST2 = False
SAVE_FILE_TEST_TRANSFORMED = False
SAVE_FILE_TEST2_TRANSFORMED = False
# SAVE_PERSPECTIVE_TRANSFORM_1 = False
# SAVE_PERSPECTIVE_TRANSFORM_2 = False


def generate_corners():
    NX = 9
    NY = 6

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)

        # Find the chessboard corners
        ret, corners = find_corners(img, NX, NY)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints


def calibrate_image(objpoints, imgpoints):
    calibration_image = cv2.imread('camera_cal/calibration1.jpg')
    mtx, dist = camera_calibration(calibration_image, objpoints, imgpoints)
    undistorted_calibration_image = undistort_image(calibration_image, mtx, dist)
    save_image(calibration_image, undistorted_calibration_image, SAVE_FILE_CALIBRATION)

    return mtx, dist


def undistort_test_images(name, save, save_transformed, mtx, dist):
    test_image = cv2.imread(name)
    undistorted_test_image = undistort_image(test_image, mtx, dist)

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    undistorted_test_image = cv2.cvtColor(undistorted_test_image, cv2.COLOR_BGR2RGB)
    save_image(test_image, undistorted_test_image, save)

    transformed_test = transform_image(undistorted_test_image)
    save_image(undistorted_test_image, transformed_test, save_transformed, True)


def perspective_transform(name, save, mtx, dist):
    img = cv2.imread(name)
    undistorted_img = undistort_image(img, mtx, dist)
    warped = warp(undistorted_img)

    undistorted_img = draw_lines(undistorted_img)
    warped = draw_lines(warped, True)

    save_image(undistorted_img, warped, save)



objpts, imgpts = generate_corners()

matrix, distortions = calibrate_image(objpts, imgpts)

undistort_test_images('test_images/test1.jpg', SAVE_FILE_TEST, SAVE_FILE_TEST_TRANSFORMED, matrix, distortions)
undistort_test_images('test_images/signs.png', SAVE_FILE_TEST2, SAVE_FILE_TEST2_TRANSFORMED, matrix, distortions)

perspective_transform('test_images/straight_lines2.jpg', SAVE_PERSPECTIVE_TRANSFORM_2, matrix, distortions)
perspective_transform('test_images/straight_lines1.jpg', SAVE_PERSPECTIVE_TRANSFORM_1, matrix, distortions)

