import cv2
import numpy as np
import matplotlib.pyplot as plt

levels = 5
#1st function
def get_laplacian_pyramid(image, levels, resize_ratio=0.5):
	pyramid = []
	img = np.float32(image)  # the image is float to avoid math issues later

	for level in range(levels - 1):
		# Applying gaussian blur to the given image
		blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
		# Down sampling the image by 2 - half of it's size
		smaller_img = cv2.resize(blurred_img, (0, 0), fx=resize_ratio, fy=resize_ratio)
		# Up sampling the image to its original size
		bigger_img = cv2.resize(smaller_img, (img.shape[1], img.shape[0]))

		laplacian = img - bigger_img  # finding the details in the laplacian layer
		pyramid.append(laplacian)  # adding the laplacian layer to the pyramid
		img = smaller_img  # moving on to the next level to compute it

	# adding the smallest image to the top of the pyramid
	pyramid.append(img)
	# and returning it
	return pyramid

#2nd function
def restore_from_pyramid(pyramidList, resize_ratio=2):
	updated_image = pyramidList[-1]  # Starting with the smallest image which is the blurriest on the top of the pyramid
	# looping from top to bottom(in the pyramid levels)
	for l in range(len(pyramidList) - 2, -1, -1):
		# resize so it can be added to the next layer in the pyramid
		updated_image = cv2.resize(updated_image, (0, 0), fx=resize_ratio, fy=resize_ratio)
		# adding it
		updated_image = cv2.add(pyramidList[l], updated_image)
	# returning the fully restored image - all of it
	return updated_image

#3rd function
def validate_operation(img):
	pyr = get_laplacian_pyramid(img, levels)
	img_restored = restore_from_pyramid(pyr)

	plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
	plt.imshow(img_restored, cmap='gray')

	plt.show()

#4th function
def blend_pyramids(levels):
	blended_pyramid = []

	for l in range(levels):
		#get each current level of the images in both pyramids
		apple_layer = pyr_apple[l]
		orange_layer = pyr_orange[l]

		#getting the size of this current level
		rows, cols = apple_layer.shape

		# Create the mask in the same size as the image
		mask = np.zeros((rows, cols), dtype=np.float32)

		mid = cols // 2 #finding the middle of an image
		start_blend = int(0.5 * cols - l)
		end_blend = int(0.5 * cols + l)

		#starting from the left side
		for c in range(start_blend):
			mask[:, c] = 1.0 #Apple side(full size)

		#The blending zone
		blend_size = end_blend - start_blend  # The width of the blending area

		#looping column by column in the blended area that we are working on
		for b in range(blend_size):
			col = start_blend + b  # The exact column we are working on currently
			mask[:, col] = 0.9 - (0.9 * b / (2 * l))

		#Blending both images using the mask
		blended_layer = orange_layer * mask + apple_layer * (1 - mask)

		#Adding this blended layer to the result pyramid
		blended_pyramid.append(blended_layer)

	return blended_pyramid


#Calls
apple = cv2.imread('apple.jpg')
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2GRAY)

orange = cv2.imread('orange.jpg')
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2GRAY)

validate_operation(apple)
validate_operation(orange)

pyr_apple = get_laplacian_pyramid(apple, levels)
pyr_orange = get_laplacian_pyramid(orange, levels)

pyr_result = []

# Your code goes here
pyr_result = blend_pyramids(levels)

## Restoring the final blended image from its Laplacian pyramid
final = restore_from_pyramid(pyr_result)
#And displaying it
plt.imshow(final, cmap='gray')
plt.show()
cv2.imwrite("result.jpg", final)
#cv2.imwrite("result_bonus.jpg", final)