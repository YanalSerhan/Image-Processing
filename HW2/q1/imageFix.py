import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_fix(image, id):
    fixed_im = ""

    if id == 1:
        fixed_im = histogram_eq(image)
    elif id == 2:
        fixed_im = gamma_correction(image, gamma=1.2)
    elif id == 3:
        fixed_im = gamma_correction(image, gamma=0.9)
    else:
        print(f"doesn't exist image with id = {id} in the directory")

    return fixed_im


def BaCS(src_image, alpha=1, beta=0):
    """
    input arguments:
    image: ndarray.
    return: image after Brightness and Contrast Stretching.
    """

    # stretching image: for each pixel, update the intensity to be: pixel * alpha + beta
    return cv2.convertScaleAbs(src_image, alpha, beta)


def gamma_correction(src_image, gamma=1.0):
    # compute 1/gamma
    inv_gamma = 1.0 / gamma

    # for each gray intensity i, store i^ (1/gamma) in table[i]
    table = np.array([(intensity / 255.0) ** inv_gamma * 255 for intensity in range(256)]).astype("uint8")

    # update the intensity of the image's pixels to the corresponding intensity in the table
    return cv2.LUT(src_image, table)


def histogram_eq(src_image):
    return cv2.equalizeHist(src_image)


def plotting_images(original_im, modified_im, name):
    original_with_text = cv2.putText(original_im.copy(), "Original", (10, 30),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    modified_with_text = cv2.putText(modified_im.copy(), name, (10, 30),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    combined = np.hstack((original_with_text, modified_with_text))

    # Show the combined image
    cv2.imshow("Image Comparison", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for i in range(1, 4):
    path = f'{i}.jpg'

    # when i = 1: the image is in png format
    if i == 1:
        path = f'{i}.png'

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    """
    # this block is for plotting the images
    # feel free to modify the parameters
    
    alpha = 1
    beta = 2
    gamma = 1.2
    
    stretched_im = BaCS(image, alpha, beta)
    gc_im = gamma_correction(image, gamma)
    eq_im = histogram_eq(image)

    # Plotting
    plotting_images(image, stretched_im, "Stretched")
    plotting_images(image, gc_im, "gamma")
    plotting_images(image, eq_im, "equalization")
    """

    fixed_image = apply_fix(image, i)
    plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray', vmin=0, vmax=255)
