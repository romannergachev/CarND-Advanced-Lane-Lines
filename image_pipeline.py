import numpy as np
import cv2
import glob
from image_correction import find_corners
from image_correction import camera_calibration
from image_correction import undistort_image
from image_correction import warp
from transforamtion_pipeline import transform_image
import matplotlib.pyplot as plt

from utils import save_image, draw_lines

SAVE = True

if SAVE:
    SAVE_FILE_CALIBRATION = 'output_images/1_undistorting.png'
    SAVE_FILE_TEST = 'output_images/2.1_undistorting_test.png'
    SAVE_FILE_TEST2 = 'output_images/2.2_undistorting_test2.png'
    SAVE_FILE_TEST_TRANSFORMED = 'output_images/3.1_transformed_test.png'
    SAVE_FILE_TEST2_TRANSFORMED = 'output_images/3.2_transformed_test2.png'
    SAVE_PERSPECTIVE_TRANSFORM_1 = 'output_images/4.1_transformed_straight_lines1.png'
    SAVE_PERSPECTIVE_TRANSFORM_1_2 = 'output_images/4.1_canny_transformed_straight_lines1.png'
    SAVE_PERSPECTIVE_TRANSFORM_2 = 'output_images/4.2_transformed_straight_lines2.png'
    SAVE_PERSPECTIVE_TRANSFORM_2_2 = 'output_images/4.2_canny_transformed_straight_lines2.png'
    SAVE_PERSPECTIVE_TRANSFORM_3 = 'output_images/4.3_transformed_test1.png'
    SAVE_PERSPECTIVE_TRANSFORM_3_2 = 'output_images/4.3_canny_transformed_test1.png'
    SAVE_PERSPECTIVE_TRANSFORM_4 = 'output_images/4.4_transformed_test5.png'
    SAVE_PERSPECTIVE_TRANSFORM_4_2 = 'output_images/4.4_canny_transformed_test5.png'
    SAVE_LANE_LINES_1 = 'output_images/5.1_draw_lanes.png'
    SAVE_LANE_LINES_2 = 'output_images/5.2_draw_lanes.png'
    SAVE_LANE_LINES_3 = 'output_images/5.3_draw_lanes.png'
    SAVE_LANE_LINES_4 = 'output_images/5.4_draw_lanes.png'
else:
    SAVE_FILE_CALIBRATION = False
    SAVE_FILE_TEST = False
    SAVE_FILE_TEST2 = False
    SAVE_FILE_TEST_TRANSFORMED = False
    SAVE_FILE_TEST2_TRANSFORMED = False
    SAVE_PERSPECTIVE_TRANSFORM_1 = False
    SAVE_PERSPECTIVE_TRANSFORM_1_2 = False
    SAVE_PERSPECTIVE_TRANSFORM_2 = False
    SAVE_PERSPECTIVE_TRANSFORM_2_2 = False
    SAVE_PERSPECTIVE_TRANSFORM_3 = False
    SAVE_PERSPECTIVE_TRANSFORM_3_2 = False
    SAVE_PERSPECTIVE_TRANSFORM_4 = False
    SAVE_PERSPECTIVE_TRANSFORM_4_2 = False
    SAVE_LANE_LINES_1 = False
    SAVE_LANE_LINES_2 = False
    SAVE_LANE_LINES_3 = False
    SAVE_LANE_LINES_4 = False


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
    test_image = cv2.imread('test_images/' + name)
    undistorted_test_image = undistort_image(test_image, mtx, dist)

    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    undistorted_test_image = cv2.cvtColor(undistorted_test_image, cv2.COLOR_BGR2RGB)
    save_image(test_image, undistorted_test_image, save)

    transformed_test = transform_image(undistorted_test_image)
    save_image(undistorted_test_image, transformed_test, save_transformed, True)


def perspective_transform(name, save, save_transformed, mtx, dist):
    img = cv2.imread('test_images/' + name)
    undistorted_img = undistort_image(img, mtx, dist)
    warped = warp(undistorted_img)

    undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    transformed = transform_image(undistorted_img)
    transformed_warped = warp(transformed)

    undistorted_img = draw_lines(undistorted_img)
    warped = draw_lines(warped, True)

    # transformed = draw_lines(transformed)
    # transformed_warped = draw_lines(transformed_warped, True)

    save_image(undistorted_img, warped, save)
    save_image(transformed, transformed_warped, save_transformed, True, True)
    return transformed_warped


def detect_line(binary_warped, save_file):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    binary_warped = binary_warped.astype(np.uint8)
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # cv2.imwrite('output_images/5_draw_lanes.png', out_img)
    # save_image(out_img, transformed_warped, save_transformed, True, True)



    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig(save_file)
    plt.close()


objpts, imgpts = generate_corners()

matrix, distortions = calibrate_image(objpts, imgpts)

undistort_test_images('test1.jpg', SAVE_FILE_TEST, SAVE_FILE_TEST_TRANSFORMED, matrix, distortions)

persp = perspective_transform('strt1.jpg', SAVE_PERSPECTIVE_TRANSFORM_2, SAVE_PERSPECTIVE_TRANSFORM_2_2, matrix, distortions)
detect_line(persp, SAVE_LANE_LINES_2)
persp = perspective_transform('strt2.jpg', SAVE_PERSPECTIVE_TRANSFORM_1, SAVE_PERSPECTIVE_TRANSFORM_1_2, matrix, distortions)
detect_line(persp, SAVE_LANE_LINES_1)
persp = perspective_transform('test1.jpg', SAVE_PERSPECTIVE_TRANSFORM_3, SAVE_PERSPECTIVE_TRANSFORM_3_2, matrix, distortions)
detect_line(persp, SAVE_LANE_LINES_3)
persp = perspective_transform('test5.jpg', SAVE_PERSPECTIVE_TRANSFORM_4, SAVE_PERSPECTIVE_TRANSFORM_4_2, matrix, distortions)
detect_line(persp, SAVE_LANE_LINES_4)
