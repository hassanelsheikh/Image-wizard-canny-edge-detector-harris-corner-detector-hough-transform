import cv2
import numpy as np
import model.image_ as image_

class imageProcessor:
    def apply_canny(self, image, kernel_size, sigma, low_threshold, high_threshold):
        # Step 1: Convert image to grayscale
        image.convertToGray()

        # Step 2: Apply Gaussian blur
        image.gaussianBlur(kernel_size, sigma)

        # Step 3: Compute gradient intensity
        gradient_mag, gradient_dir = image.gradientIntensity()

        # Step 4: Perform non-maximum suppression
        image.copyImage  = image.nonMaxSuppression(gradient_mag, gradient_dir)

        # Step 4: Perform double thresholding
        image.doubleThreshold(low_threshold, high_threshold)

        # Step 5: Perform hysteresis
        image.hysteresis()

        return image.copyImage

    def apply_harris_transform(self, image, window_size, k, threshold):

      # Step 2: Compute the Harris corner response
      image.copyImage =  image.harrisCornerDetection(k, threshold, window_size)

      return image.copyImage
    
    def apply_hough_transform(self, image, threshhold, theta_res, rho_res):
        # Step 1: Convert image to grayscale
        image.convertToGray()

        # Step 2: Apply Gaussian blur
        image.copyImage = cv2.GaussianBlur(image.copyImage, (5, 5), 0)

        # Step 3: Perform edge detection
        edges = cv2.Canny(image.copyImage, 50, 150)

        lines = image.hough_transform(edges, threshhold, theta_res, rho_res)

        image.plot_detected_lines(lines)

        return image.copyImage


        

