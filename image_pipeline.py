import cv2
from image_correction import generate_corners, calibrate_image
from image_correction import undistort_image
from image_correction import warp
from lane_detection import detect_line
from transforamtion_pipeline import transform_image


def detection_pipeline(name, mtx, dist):
    img = cv2.imread('test_images/' + name)
    temp = img.copy()
    undistorted_img = undistort_image(img, mtx, dist)

    undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

    transformed = transform_image(undistorted_img)
    transformed_warped = warp(transformed)

    detected = detect_line(transformed_warped)
    warped_back = warp(detected, True)
    masked_image = cv2.addWeighted(temp, 1, warped_back, 0.4, 0)
    cv2.imwrite('output_images/19_masked_image.png', masked_image)
    return masked_image


objpts, imgpts = generate_corners()

matrix, distortions = calibrate_image(objpts, imgpts)
persp = detection_pipeline('test5.jpg', matrix, distortions)
