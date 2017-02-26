# Use color transforms, gradients, etc., to create a thresholded binary image.

import cv2
import numpy as np


def __abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 250)):
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


def __mag_thresh(img, sobel_kernel=3, mag_thresh=(40, 100)):
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


def __color_thresholding(img, binary):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # For yellow
    yellow = cv2.inRange(hsv, (20, 100, 100), (50, 255, 255))

    # For white
    sensitivity_1 = 68
    white = cv2.inRange(hsv, (0, 0, 255 - sensitivity_1), (255, 20, 255))

    sensitivity_2 = 60
    hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    white_2 = cv2.inRange(hsl, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
    white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

    bit_layer = binary | yellow | white | white_2 | white_3

    return bit_layer


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

    direction = np.arctan2(sobel_y, sobel_x)

    mask = np.zeros_like(direction)
    mask[(direction > thresh[0]) & (direction < thresh[1])] = 1

    return mask


def transform_image(input_image, s_thresh=(100, 255), l_thresh=(120, 255)):
    """
                        Applies transformation pipeline to the initial image

    :param input_image: image to apply transformation pipeline
    :param s_thresh:    threshold for s color space
    :param l_thresh:    threshold for l color space
    :return:            transformed image
    """
    img = np.copy(input_image)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]

    gradx = __abs_sobel_thresh(l_channel, orient='x')
    grady = __abs_sobel_thresh(l_channel, orient='y')
    mag_binary = __mag_thresh(l_channel)
    dir_binary = __dir_threshold(l_channel)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    l_binary = np.zeros_like(l_channel)
    l_thresh_min = l_thresh[0]
    l_thresh_max = l_thresh[1]
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    combined_binary = np.zeros_like(combined)
    combined_binary[(combined == 1) | ((s_binary == 1) & (l_binary == 1))] = 1
    binary = combined_binary.astype(np.uint8)
    result_binary = __color_thresholding(img, binary)

    return __binary_noise_filtering(result_binary)


def __binary_noise_filtering(img, thresh=4):
    """
                   Filters noise
    :param img:    binary image
    :param thresh: threshold of neighbours
    :return:       image
    """
    k = np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
    nb_neighbours = cv2.filter2D(img, ddepth=-1, kernel=k)
    img[nb_neighbours < thresh] = 0
    return img
