import cv2
import matplotlib.pyplot as plt
from image_correction import LEFT_BOTTOM, LEFT_TOP, RIGHT_BOTTOM, RIGHT_TOP, LEFT_TOP_WARPED, RIGHT_TOP_WARPED


def save_image(image1, image2, save, gray=False, gray2=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    if gray2:
        ax1.imshow(image1, cmap='gray')
    else:
        ax1.imshow(image1)

    ax1.set_title('Original Image', fontsize=50)
    if gray:
        ax2.imshow(image2, cmap='gray')
    else:
        ax2.imshow(image2)

    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    if save:
        plt.savefig(save)
        plt.close(f)


def draw_lines(image, warped=False):
    if warped:
        left_top = LEFT_TOP_WARPED
        right_top = RIGHT_TOP_WARPED
    else:
        left_top = LEFT_TOP
        right_top = RIGHT_TOP

    cv2.line(image, LEFT_BOTTOM, left_top, (255, 0, 0), 3)
    cv2.line(image, RIGHT_BOTTOM, right_top, (255, 0, 0), 3)
    cv2.line(image, LEFT_BOTTOM, RIGHT_BOTTOM, (255, 0, 0), 3)
    cv2.line(image, left_top, right_top, (255, 0, 0), 3)

    return image
