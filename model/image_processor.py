import cv2
import numpy as np
import image

class imageProcessor:
    def __init__(self):
        pass

    def apply_canny(self, image, kernel_size=5, low_threshold=50, high_threshold=150):
        # Step 1: Convert image to grayscale
        image.convertToGray()

        # Step 2: Apply Gaussian blur
        image.gaussianBlur(kernel_size)

        # Step 3: Compute gradient intensity
        image.gradientIntensity()

        # Step 4: Perform double thresholding
        image.doubleThreshold(low_threshold, high_threshold)

        # Step 5: Perform hysteresis
        image.hysteresis()

        return image.data        