import cv2
import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import warnings
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore")


# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
    img_size = imgs_arr[0].shape
    res = []

    for img in imgs_arr:
        x = img.reshape(img_size[0] * img_size[1], 1)

        with threadpool_limits(limits=1):
            km = KMeans(n_clusters=n_colors)
            km.fit(x)

        img_compressed = km.cluster_centers_[km.labels_]
        img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

        res.append(img_compressed.reshape(img_size[0], img_size[1]))

    return np.array(res)


# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
    image_arrays = []
    lst = [file for file in os.listdir(folder) if file.endswith(formats)]
    for filename in lst:
        file_path = os.path.join(folder, filename)
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_arrays.append(gray_image)
    return np.array(image_arrays), lst


# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
    # Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values
    # will work
    x_pos = 70 + 40 * idx
    y_pos = 274
    while y_pos >= 0 and image[y_pos, x_pos] == 0:
        y_pos -= 1
    return 274 - y_pos


def calc_cum_hist(image):
    image_hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    return np.cumsum(image_hist)


def compare_hist(target_window_cum, target_cum_hist):
    emd = np.sum(np.abs(target_window_cum - target_cum_hist))
    return emd < 260


images, names = read_dir('data')
numbers, nums = read_dir('numbers')
max_student_list = []
windows_cum_hist = []
numbers_cum_hist = []
number_shape = numbers[0].shape

# sections a, b
# uncomment to run

"""
cv2.imshow(names[0],images[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
cv2.imshow(nums[0],numbers[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
plt.imshow(images[0])
plt.show()
"""

# section c
# the implementation of compare_hist is done above
# implementation details will be in the PDF file


# section d
# uncomment to run the line 121 to see the result


for j in range(len(images)):
    windows = np.lib.stride_tricks.sliding_window_view(images[j], number_shape)
    target_window = windows[113, 30]
    windows_cum_hist.append(calc_cum_hist(target_window))

for i in range(len(numbers)):
    numbers_cum_hist.append(calc_cum_hist(numbers[i]))

for j in range(len(images)):
    for i in range(len(numbers) - 1, -1, -1):
        if compare_hist(windows_cum_hist[j], numbers_cum_hist[i]):
            max_student_list.append(i)
            # print(f"the highest column in image {names[j]} contains: {i} students.")
            break

windows_cum_hist.clear()
numbers_cum_hist.clear()

# sections e, f, g
# to see section f, uncomment the print at line 145

quant_img = quantization(images, n_colors=3)

"""
_, bi_image = cv2.threshold(quant_img[0], 230, 255, cv2.THRESH_BINARY)
cv2.imshow(names[0], bi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

bar_heights = []

for im_index in range(len(images)):
    _, bi_image = cv2.threshold(quant_img[im_index], 230, 255, cv2.THRESH_BINARY)
    bar_heights.clear()
    for bar_index in range(10):
        # print(f"the bar number {bar_index} in {names[im_index]} has height: {get_bar_height(bi_image, bar_index)}")
        bar_height = get_bar_height(bi_image, bar_index)
        bar_heights.append(round(max_student_list[im_index] * bar_height / 156))

    print(f"Histogram {names[im_index]} gave {",".join(map(str, bar_heights))}")

max_student_list.clear()
bar_heights.clear()
exit()
