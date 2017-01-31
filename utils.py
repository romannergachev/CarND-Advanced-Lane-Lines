import matplotlib.pyplot as plt


def save_image(image1, image2, save, gray=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
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
    else:
        plt.show()
