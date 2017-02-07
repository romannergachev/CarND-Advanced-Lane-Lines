##Advanced Lane Finding Project

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/1_undistorting.png "Undistorted"
[image2]: ./output_images/2.2_undistorting_test2.png "Road Transformed"
[image3]: ./output_images/3.1_transformed_test.png "Binary Example"
[image4]: ./output_images/4.4_transformed_test5.png "Warp Example"
[image5]: ./output_images/5.4_draw_lanes.png "Fit Visual"
[image6]: ./output_images/9_fully_masked.png "Output"
[video1]: ./project_video_annotated.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

####1. Compute camera matrix and distortion coefficients.

The code for this step is contained in lines #24 through #90 of the file called `camera.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I used images from the camera_cal directory to get enough data for camera calibration after that I've used `objpoints` and `imgpoints` to computed the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
After that using camera matrix and dustirtion coefficients I've used `cv2.undistort()` and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
####2. Usage of color transforms, gradients or other methods to create a thresholded binary image.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #7 through #123 in `transformation_pipeline.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####3. Perspective transform with example.

The code for my perspective transform includes a function called `warp()`, which appears in lines #98 through #114 in the file `camera.py`.  The `warp()` function takes as inputs an image (`img`), as well as direction of warp (straight or inverse) (`invert`) and uses predefined `src` and `dst` constants.  
I have hardcoded the source and destination points in lines #5 to #19 `camera.py`.

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image:

![alt text][image4]

####4. Lane-line pixels identification and their fitting their using a polynomial

Then I used suggested method of finding peaks in a Histogram to search for the lane line pixels and moved over the image to detect the whole line with Sliding Windows. After that I've fitted the result to the polynomial and got the actual lane line.

You can find the basic sliding windows search in `lane.py` lines #127 to #203.

![alt text][image5]

####5. Radius of curvature of the lane and the position of the vehicle with respect to center calculation.

I did this in lines #247 through #263 `__calculate_curvature()` in my code in `lane.py` and regarding the vehicle position and center - lines #340 through #367 `add_info()` in my code in `lane.py`

####6. Result.

I implemented this steps in lines #9 through #38 in my code in `video_pipeline.py` in the function `detection_pipeline()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Result video.

Here's a [link to my video result](./project_video_annotated.mp4)

I've modified the general pipeline to call the `detect_lane()` method lines #81 to #125 (`lane.py`) that actually decides whether to use blind search (via sliding windows) or targeted search by previous known position lines #205 to #245 `__continuously_detect()`.

Also I've used `__check_line()` method (#265 to #297 same file) to check the new line is the right one.


---

###Discussion

####1. Problems and Issues

First of all, I would like to say that the current recognition algorithm should be improved (since not every line is actually recoginisable and if the road has 'fake' lines it can fail).
Second of all, from the detection perspective - it's better to implement the algorithm that would count not only the lanes itself drawn on the road (they could be missing actually), but all the other conditions:
- Road width itself (by using the standard lane size we can get the number of lanes per road and calculate our desired position)
- Positions of other vehicles on the road could help us identify the lanes
- Maybe also use data from the internet to decide on quantity of lanes on the road

In the end I would like to say that visual recognition of lanes is important, but in practice - it's not enough. We need to use combination of different methods of lane calculation.

