import numpy as np
import cv2
from scipy.ndimage import convolve


def image_1_filter(image):
    arr = np.array(image)
    filtered_img_array = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)

    # Horizontally apply the mean filter
    for y in range(arr.shape[0]):
        average_value = np.mean(arr[y, :])
        filtered_img_array[y, :] = average_value

    return filtered_img_array


def image_2_filter(image, kernel_size):
    # Define the averaging kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    # Apply convolution with the averaging kernel
    return convolve(image, kernel)


def image_3_filter(image, r=11):
    # r must be an odd number for cv2.medianBlur
    if r % 2 == 0:
        raise ValueError("Kernel size 'r' must be an odd number.")

    return cv2.medianBlur(image, r)


def image_4_filter(image, kernel_size):
    gray_image = image.copy()
    # Apply vertical average filter
    return cv2.blur(gray_image, (1, kernel_size))


def image_5_filter(image):
    return image - cv2.GaussianBlur(image, (13, 13), 4.4) + 128


def image_6_filter(image):
    arr = np.array(image)

    kernel = np.array([
        [-1.1, -1.1, -1.1],
        [0, 0, 0],
        [1.1, 1.1, 1.1]
    ])
    return cv2.filter2D(arr, -1, 0.32 * kernel)


def image_7_filter(image, original):
    height, width = image.shape[0], image.shape[1]
    upper_half = image[0:height // 2, :]
    lower_half = image[height // 2:, :]

    # Create a new image of the same size as the original image
    result = 255 * np.ones_like(original)

    # Attach the upper half of the image onto the lower half of the result image
    result[height // 2:, :] = cv2.cvtColor(upper_half, cv2.COLOR_GRAY2BGR)

    # Attach the lower half of the image onto the upper half of the result image
    result[0:height // 2, :] = cv2.cvtColor(lower_half, cv2.COLOR_GRAY2BGR)
    return result


def image_8_filter(image):
    return image


def image_9_filter(image):
    img = np.array(image)

    kernel = np.array([
        [0, -0.6, 0],
        [-0.6, 3.401, -0.6],
        [0, -0.6, 0]
    ])
    return cv2.filter2D(img, -1, kernel)


def calculate_mse(original_image, filtered_image):
    """Calculate Mean Squared Error (MSE) between original and filtered images."""
    # Ensure both images are converted to grayscale if needed
    if len(original_image.shape) == 3:  # If original image is color (3 channels)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    if len(filtered_image.shape) == 3:  # If filtered image is color (3 channels)
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # Calculate squared differences between corresponding pixels
    squared_diff = (original_image.astype(np.float32) - filtered_image.astype(np.float32)) ** 2

    # Compute the mean squared error
    mse = np.mean(squared_diff)

    return mse


original_path = "1.jpg"
# Load the original image
original_img = cv2.imread(original_path)  # Reads the image in BGR format

# Convert to grayscale
original_image_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

# Define a list of filters with their respective arguments
filters = [
    {"func": image_1_filter, "args": [original_image_gray]},
    {"func": image_2_filter, "args": [original_image_gray, 11]},
    {"func": image_3_filter, "args": [original_image_gray]},
    {"func": image_4_filter, "args": [original_image_gray, 17]},
    {"func": image_5_filter, "args": [original_image_gray]},
    {"func": image_6_filter, "args": [original_image_gray]},
    {"func": image_7_filter, "args": [original_image_gray, original_img]},
    {"func": image_8_filter, "args": [original_image_gray]},
    {"func": image_9_filter, "args": [original_image_gray]},
]

# Process the filters dynamically
for i, filter_entry in enumerate(filters, start=1):
    filtered_img = filter_entry["func"](*filter_entry["args"])
    cv2.imwrite(f"filtered_image_{i}.jpg", filtered_img)
    current_img_path = f"image_{i}.jpg"
    current_img = cv2.imread(current_img_path)
    print(f"the MSE between image_{i} and the filtered is: {calculate_mse(current_img, filtered_img)}")
