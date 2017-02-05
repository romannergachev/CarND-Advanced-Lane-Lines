import cv2
from camera import Camera
from lane import detect_line
from transforamtion_pipeline import transform_image


def detection_pipeline(img):
    temp = img.copy()

    undistorted_img = camera.undistort_image(temp)

    undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

    transformed = transform_image(undistorted_img)
    transformed_warped = camera.warp(transformed)

    detected = detect_line(transformed_warped)
    warped_back = camera.warp(detected, True)
    masked_image = cv2.addWeighted(temp, 1, warped_back, 0.4, 0)
    return masked_image


camera = Camera()
test_image = cv2.imread('test_images/' + 'test5.jpg')
persp = detection_pipeline(test_image)
