# Image Processing Coursework

This repository contains a set of image-processing assignments implemented in Python using OpenCV, NumPy, Matplotlib, and scikit-learn. The goal of the work was to learn and apply core computer-vision and image-analysis techniques, including filtering, histogram analysis, transforms, image restoration, and multi-image blending.

## Repository structure

- `HW1/`
  - `TranscribeHistogram.py` — histogram-based image transcription and student counting
  - `HW1 - answers.pdf` — assignment solution notes
- `HW2/`
  - `q1/imageFix.py` — contrast and brightness enhancement for damaged images
  - `q2/puzzleSolver.py` — image stitching and puzzle reconstruction using affine/perspective transforms
  - `HW2 report.pdf` — assignment report
- `HW3/`
  - `q1/bonus.py` — image filtering and MSE comparison for multiple filter types
  - `q2/BilateralCleaning.py` — bilateral filtering for denoising grayscale images
  - `q3/fixImg.py` — image restoration using median/Gaussian filters and averaging noisy images
  - `report.pdf` — assignment report
- `HW4/`
  - `q2/zebra.py` — Fourier-domain resizing and spectrum analysis
  - `q3/main.py` and `BadImagesFixing.py` — restoring degraded images through transform, Fourier, and filter-based techniques
  - `report.pdf` — assignment report
- `HW5/`
  - `q1/matchFaces.py` — template matching and face detection using NCC and scaling
  - `q2/blendFruits.py` and `blend2.py` — Laplacian pyramid blending of two fruit images
  - `report.pdf` — assignment report

## Technologies used

- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- scikit-learn (`KMeans`)
- SciPy (`convolve`)

---

## HW1 — Histogram-based transcription

### `HW1/TranscribeHistogram.py`

This script reads a set of images from a folder and analyzes them using cumulative histograms and K-means quantization.

What it does:
- loads grayscale images from `data` and `numbers`
- converts images to a reduced number of intensity levels using K-means clustering (`quantization`)
- computes cumulative histograms for each image and compares them against a set of reference number templates
- identifies which number is most similar to the bar pattern in each image
- uses thresholding to isolate bar regions
- estimates bar heights and prints the histogram output for each image

This project is essentially a form of visual OCR, where the code tries to interpret image content from its histogram structure.

---

## HW2 — Image enhancement and puzzle assembly

### `HW2/q1/imageFix.py`

This file focuses on fixing damaged or low-contrast images.

What it does:
- applies brightness/contrast stretching using `BaCS`
- performs gamma correction with `gamma_correction`
- uses histogram equalization with `histogram_eq`
- saves corrected versions of the images as `*_fixed.jpg`

The purpose is to improve image visibility and contrast before further analysis.

### `HW2/q2/puzzleSolver.py`

This script reconstructs puzzle pieces by aligning them using geometric transforms.

What it does:
- reads matched landmark points from `matches.txt`
- computes an affine or perspective transform matrix with `cv2.getAffineTransform` or `cv2.getPerspectiveTransform`
- warps each puzzle piece into the coordinate space of the base image
- stitches the transformed pieces together to form the final puzzle image
- saves the result as `solution.jpg`

This is a practical example of using homography/affine alignment in image stitching.

---

## HW3 — Image filtering and restoration

### `HW3/q1/bonus.py`

This file applies multiple filtering methods to one image and evaluates the result using Mean Squared Error (MSE).

What it does:
- defines custom filters for each image condition, including:
  - horizontal averaging
  - box blur
  - median blur
  - vertical smoothing
  - high-boost/sharpening
  - edge enhancement
  - row swapping and reconstruction
  - unsharp-like filtering
- applies each filter to the target image
- saves the outputs as `filtered_image_1.jpg` to `filtered_image_9.jpg`
- computes the MSE between the filtered output and the original image or target reference

This assignment was about learning how different filters affect image content and how to compare their quality numerically.

### `HW3/q2/BilateralCleaning.py`

This code performs bilateral filtering to denoise grayscale images while preserving edges.

What it does:
- extracts a neighborhood around each pixel
- computes spatial and intensity weights
- applies a bilateral filter to reduce noise while protecting edges
- runs the filter on several sample images such as `NoisyGrayImage.png`, `balls.jpg`, and `taj.jpg`
- saves denoised results as `denoised_img0.png`, `denoised_img1.png`, and `denoised_img2.png`

This is a classic edge-preserving denoising technique often used in computer vision.

### `HW3/q3/fixImg.py`

This script restores a damaged image using several smoothing operations and a noise-averaging strategy.

What it does:
- loads the broken image (`broken.jpg`)
- applies repeated median and Gaussian smoothing
- applies a bilateral filter for further cleanup
- saves the intermediate result as `semi_fixed.jpg`
- loads a stack of noisy versions of the image from `noised_images.npy`
- computes the mean image to reduce random noise
- saves the final restored version as `HQ_fix_.jpg`

This demonstrates denoising by combining repeated filtering and averaging multiple noisy observations.

---

## HW4 — Fourier-domain processing and image restoration

### `HW4/q2/zebra.py`

This script uses the Fourier transform to resize and analyze the zebra image in frequency space.

What it does:
- converts the image to grayscale
- computes the 2D FFT and visualizes the magnitude spectrum
- pads the Fourier spectrum to enlarge the image
- reconstructs the resized image using inverse FFT
- creates a version with multiple copies of the Fourier spectrum and displays the resulting reconstruction
- saves the final output as `zebra_scaled.png`

This project shows how resizing and image synthesis can be performed in the frequency domain.

### `HW4/q3/BadImagesFixing.py`

This file restores multiple degraded images using several problem-specific methods.

What it does:
- `clean_baby` — uses perspective transforms and median filtering to remove overlapping, distorted regions
- `find_peaks` — finds dominant frequency peaks in the Fourier spectrum
- `clean_windmill` — removes strong spectral peaks to reduce periodic artifacts
- `clean_watermelon` — applies a high-pass filter to sharpen the image
- `clean_umbrella` — compensates for a frequency-domain shift caused by motion or misalignment
- `clean_USAflag` — preserves the original region while smoothing the rest of the image
- `clean_house` — removes motion blur by dividing in the Fourier domain
- `clean_bears` — uses histogram equalization to improve contrast

### `HW4/q3/main.py`

This script orchestrates the restoration of the seven damaged images.

What it does:
- loads each `.tif` image from the `Images` directory
- converts it to grayscale
- applies the corresponding cleanup function
- displays original vs cleaned image side-by-side for comparison

This is a demonstration of task-specific restoration techniques for different types of image degradation.

---

## HW5 — Pattern matching and image blending

### `HW5/q1/matchFaces.py`

This file performs template matching to detect a pattern (face template) inside larger images.

What it does:
- scales images and templates using Fourier-domain resizing and bicubic interpolation
- computes the normalized cross-correlation (NCC) between the template and image windows
- suppresses duplicate matches using non-maximum suppression
- adjusts match coordinates back to the original image scale
- draws rectangles around detected matches and saves the output as `students_result.jpg` and `thecrew_result.jpg`

This is a classic image-matching workflow used for object detection and template localization.

### `HW5/q2/blendFruits.py`

This script blends two fruit images using a Laplacian pyramid.

What it does:
- computes a Laplacian pyramid for each image
- creates a blending mask across the center of the image
- mixes the pyramid levels using that mask
- reconstructs the blended image from the blended pyramid
- saves the result as `result.jpg`

This is a multi-scale blending technique that preserves detail and avoids harsh seam artifacts.

### `HW5/q2/blend2.py`

This is a second version of fruit blending using Gaussian/Laplacian pyramids and mask-based reconstruction.

What it does:
- loads `apple.jpg` and `orange.jpg`
- resizes them to the same dimensions
- creates a vertical mask that separates the left and right halves
- builds Gaussian and Laplacian pyramids for both images
- blends corresponding pyramid layers using the mask
- reconstructs the final output and saves it as `blended.jpg`

This version is a more direct implementation of pyramidal image blending.

---

## Final notes

This repository is mainly an academic image-processing project that combines:
- enhancement and restoration
- filtering and Fourier analysis
- segmentation and template matching
- geometric transformation and stitching
- multi-scale image blending

Together, the assignments provide a broad introduction to practical computer vision and digital image processing using Python.

## Running the code

Most scripts are meant to be run from their respective folders, with the relevant images present in the same directory.

Example:

```bash
cd HW1
python TranscribeHistogram.py
```

or:

```bash
cd HW5/q2
python blendFruits.py
```

## License

No explicit license file is present in the repository. Please check with the repository owner before reusing the code for publication or redistribution.
