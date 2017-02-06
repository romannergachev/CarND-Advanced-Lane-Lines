# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
import cv2
import numpy as np

# meters per pixel in y dimension
ym_per_pix = 30 / 720
# meters per pixel in x dimension
xm_per_pix = 3.7 / 700


class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
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
        self.prev_x = None
        # y values for detected line pixels
        self.ally = None
        self.prev_y = None
        self.position = None
        self.detection_counter = 0


class Lane:
    def __init__(self):
        self.shape = 720
        self.detection_margin = 100
        self.leftLine = Line()
        self.rightLine = Line()
        self.y = np.linspace(0, self.shape - 1, self.shape)
        self.out = None
        # number of images for average
        self.n = 5
        # Set minimum number of pixels found to recenter window
        self.minpix = 50

    def detect_lane(self, binary_warped):
        binary_warped = binary_warped.astype(np.uint8)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        self.out = np.zeros_like(out_img).astype(np.uint8)

        if self.leftLine.detection_counter <= self.n and self.leftLine.recent_xfitted:
            self.continuously_detect(binary_warped)
        else:
            self.initial_detect_line(binary_warped)

        if self.rightLine.detection_counter <= self.n and self.rightLine.recent_xfitted:
            self.continuously_detect(binary_warped, False)
        else:
            self.initial_detect_line(binary_warped, False)

        self.calculate_curvature()

        self.draw_lane()

        return self.out

    def initial_detect_line(self, binary_warped, is_left=True):
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

        line.recent_xfitted = []
        line.recent_xfitted.append(fitx)
        line.bestx = fitx
        line.detected = True
        line.recent_fit = []
        line.recent_fit.append(fit)
        line.best_fit = fit
        line.current_fit = fit
        line.diffs = np.array([0, 0, 0], dtype='float')
        line.allx = x
        line.ally = y
        line.detection_counter = 0

    def continuously_detect(self, binary_warped, is_left=True):
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

        line.recent_xfitted.append(fitx)
        line.bestx = np.mean(line.recent_xfitted[-self.n:], axis=0)
        line.detected = True
        line.current_fit = fit
        line.recent_fit.append(fit)
        line.best_fit = np.mean(line.recent_fit[-self.n:], axis=0)
        line.diffs = np.array([0, 0, 0], dtype='float')
        line.allx = x
        line.ally = y

        line.detection_counter = 0

    def calculate_curvature(self):
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

        self.leftLine.radius_of_curvature = left_curverad
        self.rightLine.radius_of_curvature = right_curverad

        return left_curverad, right_curverad

    def check_line(self):
        left_curverad, right_curverad = self.calculate_curvature()
        if not right_curverad - self.detection_margin < left_curverad < right_curverad + self.detection_margin:
            self.fall_back(self.leftLine)
            self.fall_back(self.rightLine)
            return False
        #add checks


    def fall_back(self, line):
        line.allx = line.prev_x
        line.ally = line.prev_y
        line.detected = False
        # add more

    def draw_lane(self):
        # Generate x and y values for plotting
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
