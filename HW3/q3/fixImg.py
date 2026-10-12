import cv2
import numpy as np

# Load the noisy image
image = cv2.imread('broken.jpg')

filtered_image = cv2.medianBlur(image, 9)  # Kernel size is 9

# Apply a Gaussian filter to further smooth the image
filtered_image = cv2.GaussianBlur(filtered_image, (5, 5), 0)

for _ in range(3):  # Apply median filter 3 times
    filtered_image = cv2.medianBlur(filtered_image, 9)

filtered_image = cv2.bilateralFilter(filtered_image, 9, 75, 75)
cv2.imwrite("semi_fixed.jpg", filtered_image)
noised_images = np.load('noised_images.npy')

mean_img = np.mean(noised_images, axis=0)
path = 'HQ_fix_.jpg'
cv2.imwrite(path, mean_img)
