import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt


def clean_baby(im):
    # Define destination points for perspective transformation - corners
    dest_location = np.float32([[0, 0], [0, 255], [255, 0], [255, 255]])

    # Define source points for each image - corners of each image
    img1_src = np.float32([[5, 19], [5, 131], [111, 19], [111, 131]])
    img2_src = np.float32([[181, 4], [121, 51], [249, 70], [177, 120]])
    img3_src = np.float32([[78, 163], [133, 244], [145, 117], [245, 160]])
    source_locations = [img1_src, img2_src, img3_src]

    # Transform source points to NumPy arrays
    source_locations = [np.array(pts, dtype=np.float32) for pts in source_locations]

    # List to store transformed images
    transformed_images = []

    # Process each source point set
    for location in source_locations:
        # Remove noise using a median filter
        denoised_image = cv2.medianBlur(im, 3)
        # Compute the perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(location, dest_location)
        # Apply the perspective transformation
        transformed_image = cv2.warpPerspective(denoised_image, transform_matrix, im.shape)
        # Add the transformed image to the list
        transformed_images.append(transformed_image)

    # Stack images and compute the median across the stack for each pixel
    transformed_stack = np.stack(transformed_images, axis=0)
    clean_baby_image = np.median(transformed_stack, axis=0).astype(np.uint8)

    return clean_baby_image


def find_peaks(image):
    # Compute the Fourier Transform and shift the zero frequency component to the center
    fft_result = np.fft.fft2(image)
    shifted_fft = np.fft.fftshift(fft_result)

    # Compute the magnitude spectrum and apply logarithmic scaling
    magnitude_spectrum = np.log1p(np.abs(shifted_fft)) * 20

    # Create a boolean mask to exclude regions near the center
    center_x, center_y = image.shape[0] // 2, image.shape[1] // 2
    mask = np.ones_like(magnitude_spectrum, dtype=bool)
    mask[center_x - 2:center_x + 3, center_y - 2:center_y + 3] = False

    # Apply the mask to filter out the center
    masked_spectrum = np.where(mask, magnitude_spectrum, -np.inf)

    # Find the top two peak indices based on the magnitude
    peak_indices = np.argpartition(masked_spectrum.ravel(), -2)[-2:]
    peak_coordinates = np.array(np.unravel_index(peak_indices, magnitude_spectrum.shape)).T

    # Sort peaks by magnitude in descending order
    sorted_peaks = sorted(
        peak_coordinates, key=lambda coord: magnitude_spectrum[tuple(coord)], reverse=True
    )

    peaks = sorted_peaks[:2]
    peaks_coor = []
    for point in peaks:
        x, y = point
        peaks_coor.append((x, y))
    # Return as a list of tuples
    return peaks_coor


def clean_windmill(im):
    # Compute the Fourier Transform and shift the zero frequency component to the center
    shifted_fourier = np.fft.fftshift(np.fft.fft2(im))

    # Define the coordinates of the peaks to be removed
    peaks = find_peaks(im)

    # Use slicing and masking to remove peaks
    mask = np.ones_like(shifted_fourier, dtype=bool)
    for x, y in peaks:
        mask[x, y] = False

    # Apply the mask to the Fourier spectrum
    cleaned_spectrum = shifted_fourier * mask

    # Perform the inverse Fourier Transform to reconstruct the image
    cleaned_image = np.abs(np.fft.ifft2(np.fft.ifftshift(cleaned_spectrum)))

    return cleaned_image


def clean_watermelon(im):
    kernel = np.array([[0, -5, 0],
                       [-5, 20, -5],
                       [0, -5, 0]], dtype=np.float32)

    # Apply the high-pass filter
    filtered_img = cv2.filter2D(im, -1, kernel)

    # Add the filtered image to the original to sharpen it
    return cv2.add(im, filtered_img)


def clean_umbrella(im):
    # Get the image's dimension
    h, w = im.shape

    # Compute the Fourier Transform of the umbrella image
    fft_img = np.fft.fft2(im)

    # Define the shift
    x0, y0 = 79, 4

    # Extract frequencies in the Fourier domain
    v_axis = np.fft.fftfreq(h)
    u_axis = np.fft.fftfreq(w)
    [u, v] = np.meshgrid(u_axis, v_axis)

    # Compute the correction factor
    # How we got to this result is explained in the pdf
    correction_factor = 1 + np.exp(-2j * np.pi * (u * x0 + v * y0))

    # To avoid dividing by small number, we set the value to 1
    correction_factor = np.where(np.abs(correction_factor) < 0.01, 1, correction_factor)
    fft_img = 2 * fft_img / correction_factor

    # Transform back to the spatial domain
    return np.abs(np.fft.ifft2(fft_img))


def clean_USAflag(im):
    # Define the region to be preserved (top-left corner)
    region_height, region_width = 90, 140
    preserved_region = im[:region_height, :region_width]

    # Create a filtered version of the image (uniform filter along the x-axis)
    kernel = np.ones((3, 37), np.float32) / (3 * 37)
    filtered_image = cv2.filter2D(im, -1, kernel)

    # Replace the top-left region of the filtered image with the original region
    filtered_image[:region_height, :region_width] = preserved_region

    return filtered_image


def clean_house(im):
    img_fourier = np.fft.fft2(im)
    # kernel for horizontal motion blur
    blur_mask = np.zeros(im.shape)

    blur_mask[0, 0:10] = 0.1

    blur_mask = np.fft.fft2(blur_mask)
    blur_mask = np.where(np.abs(blur_mask) < 0.001, 1, blur_mask)

    img_fourier = img_fourier / blur_mask
    cleaned_img = abs(np.fft.ifft2(img_fourier))
    return cleaned_img


def clean_bears(im):
    return cv2.equalizeHist(im)
