import numpy as np
import cv2
import glob
from image_correction import find_corners
from image_correction import camera_calibration
from image_correction import undistort_image
from transforamtion_pipeline import transform_image
from utils import save_image
import matplotlib.pyplot as plt
# %matplotlib qt



# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)


objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

NX = 9
NY = 6

SAVE_FILE_CALIBRATION = 'output_images/1_undistorting.png'
SAVE_FILE_TEST = 'output_images/2.1_undistorting_test.png'
SAVE_FILE_TEST2 = 'output_images/2.2_undistorting_test2.png'
SAVE_FILE_TEST_TRANSFORMED = 'output_images/3.1_transformed_test.png'
SAVE_FILE_TEST2_TRANSFORMED = 'output_images/3.2_transformed_test2.png'

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


calibration_image = cv2.imread('camera_cal/calibration1.jpg')
mtx, dist = camera_calibration(calibration_image, objpoints, imgpoints)
undistorted_calibration_image = undistort_image(calibration_image, mtx, dist)
save_image(calibration_image, undistorted_calibration_image, SAVE_FILE_CALIBRATION)

test_image = cv2.imread('test_images/test1.jpg')
undistorted_test_image = undistort_image(test_image, mtx, dist)

test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
undistorted_test_image = cv2.cvtColor(undistorted_test_image, cv2.COLOR_BGR2RGB)
save_image(test_image, undistorted_test_image, SAVE_FILE_TEST)

transformed_test = transform_image(undistorted_test_image)
save_image(undistorted_test_image, transformed_test, SAVE_FILE_TEST_TRANSFORMED, True)



test_image2 = cv2.imread('test_images/signs_vehicles_xygrad.png')
undistorted_test_image2 = undistort_image(test_image2, mtx, dist)

test_image2 = cv2.cvtColor(test_image2, cv2.COLOR_BGR2RGB)
undistorted_test_image2 = cv2.cvtColor(undistorted_test_image2, cv2.COLOR_BGR2RGB)
save_image(test_image2, undistorted_test_image2, SAVE_FILE_TEST2)

transformed_test2 = transform_image(undistorted_test_image2)
save_image(undistorted_test_image2, transformed_test2, SAVE_FILE_TEST2_TRANSFORMED, True)
