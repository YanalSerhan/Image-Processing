import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(10, 10))
plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

# Compute the 2D Fourier Transform
f_transform = fft2(image)
f_shift = fftshift(f_transform)

# Compute the magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')

# Get original dimensions
H, W = image.shape

# Step 1: Compute the 2D Fourier Transform
f_transform = fft2(image)
f_shift = fftshift(f_transform)  # Shift zero frequency to the center

# Step 2: Zero-pad in the Fourier domain to scale the image to (2H, 2W)
padded_f_shift = np.zeros((2 * H, 2 * W), dtype=complex)
center_h, center_w = H // 2, W // 2

# Place the original Fourier data at the center of the padded spectrum
padded_f_shift[H // 2:(3 * H) // 2, W // 2:(3 * W) // 2] = f_shift

# Step 3: Perform the inverse Fourier Transform
padded_f_transform = ifftshift(padded_f_shift)  # Shift back
scaled_image = ifft2(padded_f_transform).real  # Compute the inverse FFT

# Normalize the scaled image for visualization
scaled_image = np.clip(scaled_image, 0, 255).astype(np.uint8)

# Step 4: Compute the magnitude spectrum for visualization
magnitude_spectrum_padded = 20 * np.log(np.abs(padded_f_shift) + 1)

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(magnitude_spectrum_padded, cmap='gray')

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(scaled_image, cmap='gray')

# Step 1: Compute the 2D Fourier Transform
f_transform = fft2(image)
f_shift = fftshift(f_transform)  # Shift zero frequency to the center

# Get original dimensions
H, W = image.shape

# Create a new matrix to store the Fourier spectrum with four copies
four_img_copies_fourier = np.zeros((2 * H, 2 * W), np.complex128)

# Copy the values from the original Fourier spectrum to the padded matrix with zeros between every two values
four_img_copies_fourier[::2, ::2] = fftshift(fft2(image))

magnitude_spectrum = 20 * np.log(np.abs(four_img_copies_fourier) + 1)

# Reverse Fourier Transform to restore the image
four_img_copies_original = np.abs(ifft2(ifftshift(four_img_copies_fourier)))

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(four_img_copies_original, cmap='gray')

# Adjust layout to prevent overlapping
plt.tight_layout()

# call savefig before show since show clears the image
plt.savefig('zebra_scaled.png')
plt.show()


