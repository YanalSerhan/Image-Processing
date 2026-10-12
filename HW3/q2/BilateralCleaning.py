import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_window(array, center, radius):
    """
    Extracts a square window of size (2*radius+1) x (2*radius+1) around a center point in a 2D array.

    Parameters:
        array (numpy.ndarray): The input 2D array.
        center (tuple): The center point (x_c, y_c) as (row, column).
        radius (int): The radius of the window.

    Returns:
        numpy.ndarray: The extracted window.
    """
    x_c, y_c = center
    rows, cols = array.shape

    # Define the boundaries of the window
    x_min = max(0, x_c - radius)
    x_max = min(rows, x_c + radius + 1)
    y_min = max(0, y_c - radius)
    y_max = min(cols, y_c + radius + 1)

    # Extract the window
    window = array[x_min:x_max, y_min:y_max]
    return window, x_min, x_max, y_min, y_max


def compute_weights(array, center, window, x_min, x_max, y_min, y_max, std_s, std_i):
    """
    Computes the two expressions for each neighbor in the window around a center point.

    Parameters:
        array (numpy.ndarray): The input 2D array.
        center (tuple): The center point (i, j) as (row, column).
        window (numpy.ndarray): The window around the center point.
        x_min (int): The minimum row index of the window.
        x_max (int): The maximum row index of the window.
        y_min (int): The minimum column index of the window.
        y_max (int): The maximum column index of the window.
        std_s (float): The standard deviation for the spatial weights.
        std_i (float): The standard deviation for the intensity weights.

    Returns:
        numpy.ndarray, numpy.ndarray: Two arrays containing the computed values for each neighbor.
    """

    # Get the coordinates of the center point
    i, j = center

    # Get the relative coordinates in the window
    x_coords, y_coords = np.meshgrid(
        np.arange(x_min, x_max) - i,
        np.arange(y_min, y_max) - j,
        indexing='ij'
    )

    # Compute the first expression
    spatial_weights = np.exp(-(x_coords ** 2 + y_coords ** 2) / (2 * std_s ** 2))

    # Compute the second expression
    intensity_weights = np.exp(-(window - array[i, j]) ** 2 / (2 * std_i ** 2))
    return spatial_weights, intensity_weights


def clean_Gaussian_noise_bilateral(array, radius, std_s, std_i):
    """
  Applies a bilateral filter to a 2D array.

  Parameters:
      array (numpy.ndarray): The input 2D array.
      radius (int): The radius of the window.
      std_s (float): The standard deviation for the spatial weights.
      std_i (float): The standard deviation for the intensity weights.

  Returns:
      numpy.ndarray: The filtered array.
  """

    array = array.astype(np.float64)
    filtered_array = np.zeros_like(array, dtype=float)

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            center = (i, j)
            window, x_min, x_max, y_min, y_max = extract_window(array, center, radius)
            spatial_weights, intensity_weights = compute_weights(array, center, window, x_min, x_max, y_min, y_max,
                                                                 std_s, std_i)
            filtered_array[i, j] = (np.sum(spatial_weights * intensity_weights * window) - array[i][j]) / np.sum(
                spatial_weights * intensity_weights)
    return filtered_array


paths = ['NoisyGrayImage.png', 'balls.jpg', 'taj.jpg']
radiuses = [9, 3, 9]
std_s = [75, 8, 6]
std_i = [75, 25, 35]
for i,img_path in enumerate(paths):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    filtered_img = clean_Gaussian_noise_bilateral(img, radiuses[i], std_s[i], std_i[i])
    cv2.imwrite(f'denoised_img{i}.png', filtered_img)

    plt.subplot(121)
    plt.imshow(img, cmap='gray')

    plt.subplot(122)
    plt.imshow(filtered_img, cmap='gray')

    plt.show()
