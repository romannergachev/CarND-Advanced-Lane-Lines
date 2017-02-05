import cv2
import numpy as np
import glob

LEFT_BOTTOM = (245, 700)
LEFT_TOP = (600, 447)
RIGHT_TOP = (683, 447)
RIGHT_BOTTOM = (1078, 700)

LEFT_TOP_WARPED = (320, 1)
LEFT_BOTTOM_WARPED = (320, 700)
RIGHT_TOP_WARPED = (960, 1)
RIGHT_BOTTOM_WARPED = (960, 700)


class Camera:
    def __init__(self):
        self.src = np.float32([LEFT_BOTTOM, LEFT_TOP, RIGHT_TOP, RIGHT_BOTTOM])
        self.dst = np.float32([LEFT_BOTTOM_WARPED, LEFT_TOP_WARPED, RIGHT_TOP_WARPED, RIGHT_BOTTOM_WARPED])
        self.objpoints, self.imgpoints = self.generate_corners()
        self.matrix, self.distortions = self.calibrate_image()

    @staticmethod
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

    def camera_calibration(self, img):
        """
                          Returns calibrated camera matrix and distortion coefficients

        :param img:       image to get the shape
        :param objpoints: 3d points in real world space
        :param imgpoints: 2d points in image plane
        :return:          tuple of camera matrix and distortion coefficients
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

        return mtx, dist

    def undistort_image(self, img):
        """
                     Undistorts image by means of camera matrix and distortion coefficients

        :param img:  image to undistort
        :param mtx:  camera matrix
        :param dist: distortion coefficients
        :return:     undistorted image
        """
        return cv2.undistort(img, self.matrix, self.distortions, None, self.matrix)

    def warp(self, img, invert=False):
        """
                    Performs perspective transform of the image
        :param img:    image to apply perspective transform
        :param invert: check if warp should be inverted
        :return:       perspective transformed image
        """

        img_size = (img.shape[1], img.shape[0])
        if invert:
            M = cv2.getPerspectiveTransform(self.dst, self.src)
        else:
            M = cv2.getPerspectiveTransform(self.src, self.dst)

        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)

        return warped

    def generate_corners(self):
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
            ret, corners = self.find_corners(img, NX, NY)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        return objpoints, imgpoints

    def calibrate_image(self):
        calibration_image = cv2.imread('camera_cal/calibration1.jpg')
        mtx, dist = self.camera_calibration(calibration_image)

        return mtx, dist