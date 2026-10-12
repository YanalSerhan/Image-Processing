import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy.lib.stride_tricks import sliding_window_view
import warnings

warnings.filterwarnings("ignore")


def scale_down(image, resize_ratio):
    """
    Scale down an image using Fourier Transform-based optimal interpolation.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        resize_ratio (float): Scaling factor (0 < resize_ratio < 1).

    Returns:
        numpy.ndarray: Scaled-down image.
    """
    # Step 1: Apply Gaussian Blur to reduce high-frequency noise
    #sigma = 1.0 / resize_ratio  # Adjust blur intensity based on scaling factor
    blurred = cv2.GaussianBlur(image, (5, 5), 5)

    # Step 2: Compute 2D Fourier Transform
    f_transform = np.fft.fft2(blurred)
    f_shift = np.fft.fftshift(f_transform)

    # Step 3: Crop the frequency domain to retain only low frequencies
    h, w = f_shift.shape
    new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)

    # Compute cropping indices
    h_start, h_end = (h - new_h) // 2, (h + new_h) // 2
    w_start, w_end = (w - new_w) // 2, (w + new_w) // 2

    # Crop the frequency spectrum
    f_cropped = f_shift[h_start:h_end, w_start:w_end]

    # Step 4: Perform Inverse FFT to obtain the downscaled image
    f_ishift = np.fft.ifftshift(f_cropped)
    img_scaled = np.abs(np.fft.ifft2(f_ishift))

    # Normalize and return image
    img_scaled = np.uint8(cv2.normalize(img_scaled, None, 0, 255, cv2.NORM_MINMAX))

    return img_scaled


def scale_up(image, resize_ratio):
    """
    Scale up an image using Fourier Transform-based optimal interpolation.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        resize_ratio (float): Scaling factor (resize_ratio >= 1).

    Returns:
        numpy.ndarray: Scaled-up image.
    """
    # Step 1: Compute 2D Fourier Transform
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)

    # Step 2: Zero-Padding in Frequency Domain
    h, w = f_shift.shape
    new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)

    # Create a zero-padded frequency matrix
    f_padded = np.zeros((new_h, new_w), dtype=f_shift.dtype)

    # Compute insertion indices
    h_start, h_end = (new_h - h) // 2, (new_h + h) // 2
    w_start, w_end = (new_w - w) // 2, (new_w + w) // 2

    # Insert original frequency content into the center of the new matrix
    f_padded[h_start:h_end, w_start:w_end] = f_shift

    # Step 3: Perform Inverse FFT to obtain the upscaled image
    f_ishift = np.fft.ifftshift(f_padded)
    img_scaled = np.abs(np.fft.ifft2(f_ishift))

    # Normalize image and return
    img_scaled = np.uint8(cv2.normalize(img_scaled, None, 0, 255, cv2.NORM_MINMAX))
    return img_scaled


def scale_up_bicubic(image, resize_ratio):
    image_scaled = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_CUBIC)

    return image_scaled


def ncc_2d(image, pattern):
    """
    Compute the Normalized Cross-Correlation (NCC) between an image and a pattern.

    Parameters:
        image (numpy.ndarray): 2D input image.
        pattern (numpy.ndarray): 2D pattern (template).

    Returns:
        numpy.ndarray: NCC matrix with values in range [-1, 1].
    """
    image = cv2.GaussianBlur(image, (5, 5), 1)

    # Extract windows from the image with the same shape as the pattern
    windows = sliding_window_view(image, pattern.shape)  # Shape: (H', W', pattern_H, pattern_W)

    # Compute mean and standard deviation for each image window
    window_means = np.mean(windows, axis=(-2, -1), keepdims=True)
    # Compute mean and standard deviation for the pattern
    pattern_mean = np.mean(pattern)

    # Compute NCC (avoid division by zero)
    numerator = np.sum((windows - window_means) * (pattern - pattern_mean), axis=(-2, -1))
    denominator = np.sqrt(np.sum((windows - window_means) ** 2, axis=(-2, -1)) * np.sum((pattern - pattern_mean) ** 2))

    # Avoid division by zero by setting NCC to 0 where variance is 0
    # Reshape denominator to match the shape of the numerator
    denominator = denominator.squeeze()  # if denominator was originally 4D and numerator is 2D due to np.sum

    # OR if dimensions are different but not 4D and 2D:
    # denominator = denominator.reshape(numerator.shape)

    denominator[denominator == 0] = 1
    numerator[denominator == 0] = 0
    ncc = numerator / denominator

    return ncc


def display(image, pattern):
    plt.subplot(2, 3, 1)
    plt.title('Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Pattern')
    plt.imshow(pattern, cmap='gray', aspect='equal')

    ncc = ncc_2d(image, pattern)

    plt.subplot(2, 3, 5)
    plt.title('Normalized Cross-Correlation Heatmap')
    plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar()
    cbar.set_label('NCC Values')

    plt.show()


def draw_matches(image, matches, pattern_size, image_name):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for point in matches:
        y, x = point
        top_left = (int(x - pattern_size[1] / 2), int(y - pattern_size[0] / 2))
        bottom_right = (int(x + pattern_size[1] / 2), int(y + pattern_size[0] / 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)

    plt.imshow(image, cmap='gray')
    plt.show()

    cv2.imwrite(f"{image_name}_result.jpg", image)

def non_max_suppression(matches, ncc, threshold, window_size):
    filtered_matches = []
    for y, x in matches:
        if ncc[y, x] >= threshold:
            if all(abs(y - ym) > window_size[0] and abs(x - xm) > window_size[1] for ym, xm in filtered_matches):
                filtered_matches.append((y, x))
    return np.array(filtered_matches)


CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# DEMO #############
#display(image, pattern)

############# Students #############

scale_up_ratio = 1.6
scale_down_ratio = 0.79
threshold = 0.535

image_scaled = scale_up(image, scale_up_ratio)
pattern_scaled = scale_down(pattern, scale_down_ratio)

#display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = np.where(ncc >= threshold)
real_matches = np.stack((real_matches[0], real_matches[1]), axis=-1)
real_matches = non_max_suppression(real_matches, ncc, threshold=0.49, window_size=(3, 3))

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:, 0] += pattern_scaled.shape[0] // 2  # if pattern was not scaled, replace this with "pattern"
real_matches[:, 1] += pattern_scaled.shape[1] // 2  # if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.
inverse_resize_ratio = 1 / scale_up_ratio
real_matches = real_matches.astype(np.float64)  # Change the dtype of real_matches to float64
real_matches[:, 0] *= inverse_resize_ratio
real_matches[:, 1] *= inverse_resize_ratio
real_matches = real_matches.astype(np.int64)  # Change the dtype back to int64 for drawing matches

draw_matches(image, real_matches, pattern_scaled.shape, "students")  # if pattern was not scaled, replace this with
# "pattern"

############# Crew #############

curr_image = "thecrew"

image = cv2.imread(f'{curr_image}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

############# DEMO #############
#display(image, pattern)

scale_up_ratio = 2.52
scale_down_ratio = 0.58
threshold = 0.51

image_scaled = scale_up_bicubic(image, scale_up_ratio)
pattern_scaled = scale_down(pattern, scale_down_ratio)

#display(image_scaled, pattern_scaled)

ncc = ncc_2d(image_scaled, pattern_scaled)
real_matches = np.where(ncc >= threshold)
real_matches = np.stack((real_matches[0], real_matches[1]), axis=-1)
#real_matches = non_max_suppression(real_matches, ncc, threshold=0.4, window_size=(1, 1))

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:, 0] += pattern_scaled.shape[0] // 2  # if pattern was not scaled, replace this with "pattern"
real_matches[:, 1] += pattern_scaled.shape[1] // 2  # if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

inverse_resize_ratio = 1 / scale_up_ratio
real_matches = real_matches.astype(np.float64)  # Change the dtype of real_matches to float64
real_matches[:, 0] *= inverse_resize_ratio
real_matches[:, 1] *= inverse_resize_ratio
real_matches = real_matches.astype(np.int64)  # Change the dtype back to int64 for drawing matches

draw_matches(image, real_matches, pattern_scaled.shape, "thecrew")  # if pattern was not scaled, replace this with "pattern"
