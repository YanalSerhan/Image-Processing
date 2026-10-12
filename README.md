# Image Processing Coursework

This repository contains a collection of image-processing and computer-vision assignments completed in Python. The projects focus on practical concepts such as histogram analysis, image transformation, grayscale processing, clustering-based quantization, and image comparison techniques.

## Repository structure

- `HW1/` - Histogram-based image transcription and analysis
  - `TranscribeHistogram.py` - Python script for histogram-based quantization and bar counting
  - `HW1 - answers.pdf` - assignment answers/report
- `HW2/` - Homework 2 assignment materials
  - `q1/`, `q2/` - question-specific files
  - `HW2 report.pdf`
- `HW3/` - Homework 3 assignment materials
  - `q1/`, `q2/`, `q3/` - question-specific files
  - `report.pdf`
- `HW4/` - Homework 4 assignment materials
  - `q2/`, `q3/` - question-specific files
  - `report.pdf`
- `HW5/` - Homework 5 assignment materials
  - `q1/`, `q2/` - question-specific files
  - `report.pdf`

## Main technologies used

- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- scikit-learn (`KMeans`)
- threadpoolctl

## Example project

The script in `HW1/TranscribeHistogram.py` demonstrates several image-processing ideas:

- converting images to grayscale
- reading batches of images from directories
- performing color quantization using K-means clustering
- calculating cumulative histograms
- thresholding binary images
- estimating bar heights and extracting information from image histograms

This is a good example of using statistical image analysis to infer structured data from visual input.

## Getting started

Clone the repository:

```bash
git clone https://github.com/YanalSerhan/Image-Processing.git
cd Image-Processing
```

Install the required dependencies:

```bash
pip install opencv-python numpy matplotlib scikit-learn threadpoolctl
```

Run a script, for example:

```bash
python HW1/TranscribeHistogram.py
```

Note: Some scripts may rely on local datasets or folders such as `data/` and `numbers/` being present in the same working directory.

## Notes

- This repository is primarily academic/coursework-oriented.
- Many folders include assignment reports in PDF format, which provide the full context for each exercise.
- The code is intended for learning and experimentation in digital image processing.

## License

No explicit license file is included in the repository, so please check with the repository owner before reusing or redistributing the code or PDFs beyond personal study.
