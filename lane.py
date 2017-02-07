import cv2
import numpy as np

# meters per pixel in y dimension
ym_per_pix = 30 / 720
# meters per pixel in x dimension
xm_per_pix = 3.7 / 700


class Line:
    """
    Class contains info about line
    """
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the current fit of the line
        self.current_xfitted = None
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients over the last iterations
        self.recent_fit = []
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # previous x values for detected line pixels
        self.prev_x = None
        # y values for detected line pixels
        self.ally = None
        # previous y values for detected line pixels
        self.prev_y = None
        # line position
        self.position = None
        # previous line position
        self.prev_position = None
        # number of times line was failed to detect
        self.detection_counter = 0


class Lane:
    """
    Class contains info about lane and consists of two lines
    """
    def __init__(self):
        # height of the image
        self.shape = 720
        # margin
        self.detection_margin = 100
        # left line of the lane
        self.leftLine = Line()
        # right line of the lane
        self.rightLine = Line()
        # y linespace for fitted data
        self.y = np.linspace(0, self.shape - 1, self.shape)
        # output image
        self.out = None
        # number of images for average
        self.n = 5
        # number of failed images before starting the blind search
        self.n_skip = 15
        # Set minimum number of pixels found to recenter window
        self.minpix = 50
        # Width of the lane
        self.width = 0
        # frame number
        self.frame = 0
        # square error margin
        self.error_margin = 2000000

    def detect_lane(self, binary_warped):
        """
                              Detects the lane on the image and returns it as a mask image
        :param binary_warped: image to detect lane
        :return:              empty image with drawn lane on it
        """
        binary_warped = binary_warped.astype(np.uint8)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        self.out = np.zeros_like(out_img).astype(np.uint8)

        self.frame += 1

        # left line detection
        if self.leftLine.detection_counter <= self.n_skip and self.leftLine.recent_xfitted:
            # targeted search
            init_left = False
            self.__continuously_detect(binary_warped)
        else:
            # blind search
            init_left = True
            self.__initial_detect_line(binary_warped)

        # right line detection
        if self.rightLine.detection_counter <= self.n_skip and self.rightLine.recent_xfitted:
            # targeted search
            init_right = False
            self.__continuously_detect(binary_warped, False)
        else:
            # blind search
            init_right = True
            self.__initial_detect_line(binary_warped, False)

        if self.frame == 1:
            self.width = self.rightLine.position - self.leftLine.position

        # calculate curvature
        left_curverad, right_curverad = self.__calculate_curvature()

        # check lines to "look like" correct ones
        self.__check_line(self.leftLine, init_left, left_curverad)
        self.__check_line(self.rightLine, init_right, right_curverad)

        self.__draw_lane()

        return self.out

    def __initial_detect_line(self, binary_warped, is_left=True):
        """
                              Blindly search for the lane line
        :param binary_warped: image to find a line
        :param is_left:       True if it is a left line
        """
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)

        if is_left:
            line = self.leftLine
            x_base = np.argmax(histogram[:midpoint])
        else:
            line = self.rightLine
            x_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        x_current = x_base
        # Create empty lists to receive lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_x_low = x_current - self.detection_margin
            win_x_high = x_current + self.detection_margin
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (
                nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the list
            lane_inds.append(good_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > self.minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the array of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        line.position = x_base

        # Fit a second order polynomial
        fit = np.polyfit(y, x, 2)

        fitx = fit[0] * self.y ** 2 + fit[1] * self.y + fit[2]

        base_x = fit[0] * (self.shape - 1) ** 2 + fit[1] * (self.shape - 1) + fit[2]

        # fill in line params
        line.recent_xfitted = []
        line.position = base_x
        line.recent_xfitted.append(fitx)
        line.bestx = fitx
        line.current_xfitted = fitx
        line.detected = True
        line.recent_fit = []
        line.recent_fit.append(fit)
        line.best_fit = fit
        line.current_fit = fit
        line.diffs = np.array([0, 0, 0], dtype='float')
        line.allx = x
        line.ally = y
        line.detection_counter = 0

    def __continuously_detect(self, binary_warped, is_left=True):
        """
                              Targeted lane line search
        :param binary_warped: image to find a line
        :param is_left:       True if it is a left line
        """
        if is_left:
            line = self.leftLine
        else:
            line = self.rightLine

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lane_inds = ((nonzerox > (line.best_fit[0] * (nonzeroy ** 2) + line.best_fit[1] * nonzeroy + line.best_fit[2] - self.detection_margin))
                     & (nonzerox < (line.best_fit[0] * (nonzeroy ** 2) + line.best_fit[1] * nonzeroy + line.best_fit[2] + self.detection_margin)))

        # Again, extract line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        # Fit a second order polynomial to each
        fit = np.polyfit(y, x, 2)
        # Generate x and y values for plotting
        fitx = fit[0] * self.y ** 2 + fit[1] * self.y + fit[2]

        base_x = fit[0] * (self.shape - 1) ** 2 + fit[1] * (self.shape - 1) + fit[2]

        # fill in line params
        line.current_xfitted = fitx
        line.current_fit = fit
        line.diffs = np.absolute(np.subtract(line.best_fit, fit))
        line.prev_x = line.allx
        line.prev_y = line.ally
        line.allx = x
        line.ally = y
        line.prev_position = line.position
        line.position = base_x

    def __calculate_curvature(self):
        """
                 Calculates curvature of the lines
        :return: left and right lines curvature
        """
        y_eval_left = np.max(self.leftLine.ally)
        y_eval_right = np.max(self.rightLine.ally)
        # allx for right and left lines should be averaged (best coeffs)
        left_fit_cr = np.polyfit(self.leftLine.ally * ym_per_pix, self.leftLine.allx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.rightLine.ally * ym_per_pix, self.rightLine.allx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval_left * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) \
            / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval_right * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) \
            / np.absolute(2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def __check_line(self, line, init, curvature):
        """
                          Checks the line to be correct
        :param line:      line to check
        :param init:      True if it is init step
        :param curvature: lines' curvature
        :return:          True if line is confirmed
        """
        if init:
            line.radius_of_curvature = curvature
            return True

        # coefficients check
        coeffs_difference = line.best_fit - line.current_fit
        delta = coeffs_difference[0] * self.y ** 2 + coeffs_difference[1] * self.y + coeffs_difference[2]
        squared_error = np.sum(np.power(delta, 2))

        if squared_error > self.error_margin:
            print("Fall back coeffs")
            self.__fall_back(line)
            return False

        # lane width check
        difference = self.rightLine.position - self.leftLine.position
        if not (self.width - self.detection_margin * 2 < difference < self.width + self.detection_margin * 2):
            print("Fall back width")
            self.__fall_back(line)
            return False

        self.__detected(line)
        line.radius_of_curvature = curvature

        return True

    def __fall_back(self, line):
        """
                     Falls back to the previous detected line data
        :param line: left or right line
        """
        line.allx = line.prev_x
        line.ally = line.prev_y
        line.detected = False
        line.position = line.prev_position
        line.detection_counter += 1

    def __detected(self, line):
        """
                     Update the recognized line data
        :param line: left or right line
        """
        line.recent_xfitted.append(line.current_xfitted)
        line.bestx = np.mean(line.recent_xfitted[-self.n:], axis=0)
        line.detected = True
        line.recent_fit.append(line.current_fit)
        line.best_fit = np.mean(line.recent_fit[-self.n:], axis=0)
        line.detection_counter = 0
        line.position = line.best_fit[0] * (self.shape - 1) ** 2 + line.best_fit[1] * (self.shape - 1) + line.best_fit[2]

    def __draw_lane(self):
        """
                Draws lane
        """
        left_fitx = self.leftLine.best_fit[0] * self.y ** 2 + self.leftLine.best_fit[1] * self.y + \
                    self.leftLine.best_fit[2]
        right_fitx = self.rightLine.best_fit[0] * self.y ** 2 + self.rightLine.best_fit[1] * self.y + \
                     self.rightLine.best_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.y])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.polylines(self.out, np.int_([pts_left]), False, (0, 0, 255), thickness=30)
        cv2.polylines(self.out, np.int_([pts_right]), False, (255, 0, 0), thickness=30)
        cv2.fillPoly(self.out, np.int_([pts]), (0, 255, 0))

    def add_info(self, masked_image):
        """
                             Draws radius, position for each line and cars' location on the lane
        :param masked_image: image to draw the info on
        :return:             updated image
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        left_radius = "Left R = {0:.1f}m".format(self.leftLine.radius_of_curvature)
        cv2.putText(masked_image, left_radius, (20, 640), font, 1, (255, 255, 255), 2)
        right_radius = "Right R = {0:.1f}m".format(self.rightLine.radius_of_curvature)
        cv2.putText(masked_image, right_radius, (950, 640), font, 1, (255, 255, 255), 2)

        # Write the x coords for each lane
        left_line_position = "X = {0:.1f}".format(self.leftLine.position)
        cv2.putText(masked_image, left_line_position, (20, 680), font, 1, (255, 255, 255), 2)
        right_line_position = "X = {0:.1f}".format(self.rightLine.position)
        cv2.putText(masked_image, right_line_position, (950, 680), font, 1, (255, 255, 255), 2)

        # Write dist from center
        center = 1280 / 2.
        lane_width = self.rightLine.position - self.leftLine.position
        center_x = lane_width / 2.0 + self.leftLine.position
        cms_per_pixel = 370.0 / lane_width
        dist_from_center = (center_x - center) * cms_per_pixel / 100
        dist_text = "Distance from Center = {0:.1f}m".format(dist_from_center)
        cv2.putText(masked_image, dist_text, (450, 50), font, 1, (255, 255, 255), 2)

        return masked_image
