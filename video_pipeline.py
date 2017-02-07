from moviepy.editor import VideoFileClip
import imageio
import cv2
from camera import Camera
from lane import Lane
from transforamtion_pipeline import transform_image


def detection_pipeline(img):
    """
                Searches for the lanes on the image, highlights lane and draws info
    :param img: image to draw a mask on
    :return:    masked image
    """
    temp = img.copy()

    # to_write = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('to_test/TEST' + str(lane.frame) + '.png', to_write)

    # undistorted image
    undistorted_img = camera.undistort_image(temp)
    # change to RGB
    undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
    # warp image
    warped = camera.warp(undistorted_img)
    # apply filters to the image
    transformed_warped = transform_image(warped)
    # detect lines on the image
    detected = lane.detect_lane(transformed_warped)
    # warp detected lines back
    warped_back = camera.warp(detected, True)
    # apply detected lines to the initial image
    masked_image = cv2.addWeighted(temp, 1, warped_back, 0.4, 0)
    # add curvature and location on the image
    masked_image = lane.add_info(masked_image)

    # cv2.imwrite('to_test/TTTEST.png', masked_image)
    return masked_image

imageio.plugins.ffmpeg.download()
camera = Camera()
lane = Lane()

white_output = 'project_video_annotated.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(detection_pipeline)
white_clip.write_videofile(white_output, audio=False)
