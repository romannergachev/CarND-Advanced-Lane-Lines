import cv2
from camera import Camera
from lane import Lane
from transforamtion_pipeline import transform_image


def detection_pipeline(img):
    temp = img.copy()

    undistorted_img = camera.undistort_image(temp)

    undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)

    transformed = transform_image(undistorted_img)
    transformed_warped = camera.warp(transformed)

    detected = lane.detect_lane(transformed_warped)
    warped_back = camera.warp(detected, True)
    masked_image = cv2.addWeighted(temp, 1, warped_back, 0.4, 0)

    # Write the radius of curvature for each lane
    font = cv2.FONT_HERSHEY_SIMPLEX
    left_roc = "Roc: {0:.2f}m".format(lane.leftLine.radius_of_curvature)
    cv2.putText(masked_image, left_roc, (10, 650), font, 1, (255, 255, 255), 2)
    right_roc = "Roc: {0:.2f}m".format(lane.rightLine.radius_of_curvature)
    cv2.putText(masked_image, right_roc, (1020, 650), font, 1, (255, 255, 255), 2)

    # Write the x coords for each lane
    left_coord = "X  : {0:.2f}".format(lane.leftLine.position)
    cv2.putText(masked_image, left_coord, (10, 700), font, 1, (255, 255, 255), 2)
    right_coord = "X  : {0:.2f}".format(lane.rightLine.position)
    cv2.putText(masked_image, right_coord, (1020, 700), font, 1, (255, 255, 255), 2)

    # Write dist from center
    perfect_center = 1280 / 2.
    lane_x = lane.rightLine.position - lane.leftLine.position
    center_x = (lane_x / 2.0) + lane.leftLine.position
    cms_per_pixel = 370.0 / lane_x  # US regulation lane width = 3.7m
    dist_from_center = (center_x - perfect_center) * cms_per_pixel
    dist_text = "Dist from Center: {0:.2f} cms".format(dist_from_center)
    cv2.putText(masked_image, dist_text, (450, 50), font, 1, (255, 255, 255), 2)

    # cv2.imwrite('test/TEST' + str(lane.frame) + '.png', masked_image)
    return masked_image

lane = Lane()
camera = Camera()
test_image = cv2.imread('test_images/' + 'test5.jpg')
# persp = detection_pipeline(test_image)
# persp = detection_pipeline(test_image)
# persp = detection_pipeline(test_image)
