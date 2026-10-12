import cv2
import numpy as np

# Load images
img1 = cv2.imread('apple.jpg')
img2 = cv2.imread('orange.jpg')


# Ensure images are the same size
rows, cols = img1.shape[:2]
img2 = cv2.resize(img2, (cols, rows))

# Create a Sharper Mask Instead of a Gradient
mask = np.zeros((rows, cols), dtype=np.uint8)
mask[:, cols//2:] = 255  # Apple on the right, Orange left

# Normalize mask (convert 0-255 to 0-1)
mask = mask.astype(np.float32) / 255.0

# Function to Generate Gaussian & Laplacian Pyramids
def generate_pyramids(image, levels=5):  # Reduced to 5 levels for better detail
    gp = [image]
    for _ in range(levels):
        image = cv2.pyrDown(image)
        gp.append(image)

    lp = [gp[-1]]  # Smallest Gaussian image
    for i in range(levels, 0, -1):
        expanded = cv2.pyrUp(gp[i], dstsize=(gp[i-1].shape[1], gp[i-1].shape[0]))
        laplacian = cv2.subtract(gp[i-1], expanded)
        lp.append(laplacian)
    return gp, lp

# Generate pyramids
gp1, lp1 = generate_pyramids(img1)
gp2, lp2 = generate_pyramids(img2)

# Ensure Gaussian Mask Pyramid Matches Laplacian Pyramid
gp_mask = [mask]
for _ in range(len(lp1) - 1):
    mask = cv2.pyrDown(mask)
    gp_mask.append(mask)

# Blend Laplacian Pyramids Using the Mask Pyramid
lp_blend = []
for l1, l2, gm in zip(lp1, lp2, gp_mask):
    gm_resized = cv2.resize(gm, (l1.shape[1], l1.shape[0]))  # Ensure size matches
    gm_resized = gm_resized[:, :, np.newaxis]  # Reshape for broadcasting

    l_blend = l1 * gm_resized + l2 * (1 - gm_resized)
    lp_blend.append(l_blend)

# Reconstruct Final Blended Image
blended = lp_blend[0]
for i in range(1, len(lp_blend)):
    blended = cv2.pyrUp(blended, dstsize=(lp_blend[i].shape[1], lp_blend[i].shape[0]))
    blended = cv2.add(blended, lp_blend[i])

# Normalize the blended image
blended = cv2.normalize(blended, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# Convert to uint8 (grayscale image should be 0-255)
blended = np.clip(blended, 0, 255).astype(np.uint8)
blended = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)

# Save and Display Result
cv2.imwrite('blended.jpg', blended)
cv2.imshow("blended", blended)
cv2.waitKey(0)
cv2.destroyAllWindows()
