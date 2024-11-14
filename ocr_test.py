from utils import *
import cv2
import numpy as np  
import matplotlib.pyplot as plt
from PIL import Image


input_image = cv2.imread('Images/IMG_8248.jpg')

# Convert to grayscale
gray = to_grayscale(input_image)

# # Apply CLAHE to enhance contrast
enhanced_contrast = clahe(gray)

# Denoise the image
# quiet = gaussian_blur(enhanced_contrast)
quiet = median_blur(enhanced_contrast, kernel_size=3)

text = pytesseract.image_to_string(quiet)
print(text)

cv2.imshow("Image", quiet)
cv2.imshow("Output", gray)

# # Apply adaptive thresholding
# adaptive_bi = adaptive_threshold(quiet)

# # # Reduce noise with Median Blurring
# # smoothed = median_blur(adaptive_bi, kernel_size=3)

# # Further clean up with Morphological Opening
# cleaned_image = morphological_opening(adaptive_bi, kernel_size=(3, 3))

# # Display results
# plt.figure(figsize=(12, 8))

# # Original Image
# plt.subplot(1, 5, 1)
# plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# # Contrast Enhanced
# plt.subplot(1, 5, 2)
# plt.imshow(enhanced_contrast, cmap='gray')
# plt.title('Contrast Enhanced')
# plt.axis('off')

# # Denoised Image
# plt.subplot(1, 5, 3)
# plt.imshow(quiet, cmap='gray')
# plt.title('Denoised')
# plt.axis('off')

# # Adaptive Thresholded
# plt.subplot(1, 5, 4)
# plt.imshow(adaptive_bi, cmap='gray')
# plt.title('Adaptive Thresholded')
# plt.axis('off')

# # Cleaned Image
# plt.subplot(1, 5, 5)
# plt.imshow(cleaned_image, cmap='gray')
# plt.title('Noise Reduced')
# plt.axis('off')

# plt.tight_layout()
# plt.show()