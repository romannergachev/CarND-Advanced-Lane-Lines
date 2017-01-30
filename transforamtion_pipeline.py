# Use color transforms, gradients, etc., to create a thresholded binary image.

import cv2
import numpy as np


def __abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 80)):
    """
                         Performs sobel transform of the image

    :param img:          image to apply the sobel
    :param orient:       x or y orientation of sobel
    :param sobel_kernel: size of the sobel kernel
    :param thresh:       threshold for sobel
    :return:             transformed image
    """

    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    scaled_sobelx = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mask_binary = np.zeros_like(scaled_sobelx)
    mask_binary[(scaled_sobelx >= thresh[0]) & (scaled_sobelx <= thresh[1])] = 1

    return mask_binary


def __mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    """
                         Performs sobel magnitude transform

    :param img:          image to apply the sobel magnitude
    :param sobel_kernel: size of the sobel kernel
    :param mag_thresh:   threshold for sobel magnitude
    :return:             transformed image
    """

    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel_xy = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    scaled = np.uint8(255 * sobel_xy / np.max(sobel_xy))

    mask = np.zeros_like(scaled)
    mask[(scaled > mag_thresh[0]) & (scaled < mag_thresh[1])] = 1

    return mask


def __dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3)):
    """
                         Performs sobel direction transform

    :param img:          image to apply the sobel direction
    :param sobel_kernel: size of the sobel kernel
    :param thresh:       threshold for sobel direction
    :return:             transformed image
    """

    sobel_x = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobel_y = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(sobel_y, sobel_x)
    # 5) Create a binary mask where direction thresholds are met
    mask = np.zeros_like(direction)
    mask[(direction > thresh[0]) & (direction < thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    binary_output = mask  # Remove this line
    return binary_output


def transform_image(img, s_thresh=(180, 240), sx_thresh=(20, 60)):
    """
                      Applies transformation pipeline to the initial image

    :param img:       image to apply transformation pipeline
    :param s_thresh:  threshold for s color space
    :param sx_thresh: threshold for sobel related transformations
    :return:          transformed image
    """
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    gradx = __abs_sobel_thresh(l_channel, orient='x')
    grady = __abs_sobel_thresh(l_channel, orient='y')
    mag_binary = __mag_thresh(l_channel)
    dir_binary = __dir_threshold(l_channel)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(combined), combined, s_binary))
    return color_binary
