import cv2
import numpy as np
import model.image_ as image_

class imageProcessor:
    def apply_canny(self, image, kernel_size, low_threshold, high_threshold):
        # Step 1: Convert image to grayscale
        image.convertToGray()

        # Step 2: Apply Gaussian blur
        image.gaussianBlur(kernel_size)

        # Step 3: Compute gradient intensity
        gradient_mag, gradient_dir = image.gradientIntensity()

        # Step 4: Perform non-maximum suppression
        image.copyImage  = image.nonMaxSuppression(gradient_mag, gradient_dir)

        # Step 4: Perform double thresholding
        image.doubleThreshold(low_threshold, high_threshold)

        # Step 5: Perform hysteresis
        image.hysteresis()

        return image.copyImage

               