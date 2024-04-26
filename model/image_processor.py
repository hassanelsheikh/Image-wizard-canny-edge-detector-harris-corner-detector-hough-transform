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

    def apply_hough_transform(self):
    # Call the Hough transform method and plot detected lines
      gray_image = self.image_processor.rgb_to_grayscale(self.image.data)

     # Perform edge detection
      edges = self.image.edge_detection(threshold=self.high_threshold)

     # Perform Hough transform and detect lines
      accumulator, thetas, rhos = self.image.hough_transform(edges, self.theta_res, self.rho_res)

     # Plot detected lines using the ImageProcessor class
      self.image_processor.plot_detected_lines(gray_image, accumulator, rhos, thetas, self.threshold_ratio)           

    def apply_harris_transform(self, image, window_size, k, threshold):

      # Step 2: Compute the Harris corner response
      image.copyImage =  image.harrisCornerDetection(k, threshold, window_size)

      return image.copyImage