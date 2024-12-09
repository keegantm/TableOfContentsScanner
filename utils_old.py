# OCR Preprocessing Tools

import cv2
import numpy as np 
import pytesseract
from PIL import Image


# Grayscale Conversion
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

'''Noise Reduction Tools'''

# Gaussian Blurring
def gaussian_blur(image, kernel_size=(5,5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Median Blurring
def median_blur(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

'''Thresholding Functions'''
# OTSU Thresholding
def otsu_threshold(image):
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

# Adaptive Thresholding
def adaptive_threshold(image, block_size=11, C=2):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

# Descewing
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Dilation
def dilate(image, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.dilate(image, kernel, iterations=1)

# Erosion
def erode(image, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.erode(image, kernel, iterations=1)

'''Edge Detection and Contours'''
# Canny Edge Detection
def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

# Find Contours
def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

'''Morphology'''
# Opening
def morphological_opening(image, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing
def morphological_closing(image, kernel_size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Text Line and Word Segmentation
def connected_components(image):
    num_labels, labels = cv2.connectedComponents(image)
    return num_labels, labels

'''Contrast Adjustments'''

# Histogram equalization
def histogram_equalization(image):
    return cv2.equalizeHist(image)

# CLAHE
def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)
