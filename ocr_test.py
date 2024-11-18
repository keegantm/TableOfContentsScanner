import cv2
import pytesseract
import matplotlib.pyplot as plt
from utils import *

# Load the original image
input_image = cv2.imread('Images/IMG_8248.jpg')

# Convert to grayscale
gray = to_grayscale(input_image)

# Apply CLAHE for localized contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_contrast = clahe.apply(gray)

# Reduce noise with bilateral filter
quiet = cv2.bilateralFilter(enhanced_contrast, d=9, sigmaColor=75, sigmaSpace=75)

# Adaptive thresholding for binarization
adaptive_bi = cv2.adaptiveThreshold(quiet, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Optional: Dilate to make text more readable
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
processed = cv2.dilate(adaptive_bi, kernel, iterations=2)

# Display the final processed image
plt.figure(figsize=(6, 6))
plt.imshow(processed, cmap='gray')
plt.title("Processed Image for OCR")
plt.axis('off')
plt.show()

# Extract text using Tesseract
config = '--oem 1 --psm 6'
text = pytesseract.image_to_string(processed, config=config)
print("Extracted Text:\n", text)
