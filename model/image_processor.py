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
    def apply_hough(self, image, rho, theta, threshold):
        from model.hough_line_detector import apply_hough  # Importing here to avoid circular imports
        edges = image.copyImage  # Assuming edge image is ready, you might need to prepare it
        lines = apply_hough(edges, rho, theta, threshold)
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image.data, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return image.data

    def apply_harris(self, image, threshold):
        from model.harris_corner_detector import apply_harris  # Importing here to avoid circular imports
        corners = apply_harris(image.copyImage, threshold=threshold)
        image.data[corners] = [0, 0, 255]
        return image.data  
               