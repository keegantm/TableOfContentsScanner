from utils import *
import cv2
import numpy as np  
import matplotlib.pyplot as plt

input_image = cv2.imread('Images/IMG_8248.jpg')

# Convert it to grayscale
gray = to_grayscale(input_image)

# Binarize the image
otsu_bi = otsu_threshold(gray)
adapted_bi = adaptive_threshold(gray)

# Noise removal

# Display image