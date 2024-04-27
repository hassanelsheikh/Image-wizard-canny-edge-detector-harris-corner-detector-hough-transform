import cv2
import numpy as np
import matplotlib.pyplot as plt


class Image:
    def __init__(self, data, width, height):
        self.data = data
        self.width = width
        self.height = height
        self.strongEdges = None
        self.weakEdges = None
        self.copyImage = None

    def read(self, path):
        self.data = cv2.imread(path)
        self.width = self.data.shape[1]
        self.height = self.data.shape[0]
        self.copyImage = self.data.copy()

    def resize(self, width, height):
        self.data = cv2.resize(self.data, (width, height))
        self.copyImage = cv2.resize(self.copyImage, (width, height))
        self.width = width
        self.height = height

    def display(self):
        cv2.imshow('Image', self.data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convertToGray(self):
        self.copyImage = cv2.cvtColor(self.copyImage, cv2.COLOR_BGR2GRAY)


        # def gaussianBlur(self, kernel_size, sigma):
        # # Generate Gaussian kernel
        # kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2
        # kernel /= kernel_size**2
        # self.copyImage = cv2.GaussianBlur(self.copyImage, (kernel_size, kernel_size), sigma)
        # # Convolve the image with the kernel
        # self.copyImage = cv2.filter2D(self.copyImage, -1, kernel)
    def gaussianBlur(self, kernel_size, sigma):
        # Generate Gaussian kernel
        kernel = self.createGaussianKernel(kernel_size, sigma)
        # Convolve the image with the kernel
        self.copyImage = cv2.filter2D(self.copyImage, -1, kernel)
        

    def createGaussianKernel(self, kernel_size, sigma):
        kernel = np.zeros((kernel_size, kernel_size), np.float32)
        center = kernel_size // 2

        total_sum = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
                total_sum += kernel[i, j]

        # Normalize the kernel
        kernel /= total_sum
        return kernel

    def gradientIntensity(self):
        # Sobel kernels
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        # Convolve the image with the kernels
        gradient_x = cv2.filter2D(self.copyImage, -1, kernel_x)
        gradient_y = cv2.filter2D(self.copyImage, -1, kernel_y)

        # Compute the gradient intensity
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # Normalize the gradient intensity
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

        #gradient_direction
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        return gradient_magnitude, gradient_direction
    
    def nonMaxSuppression(self, gradient_magnitude, gradient_direction):
        suppressed_image = np.zeros_like(gradient_magnitude)
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                direction = gradient_direction[i, j]
                if (0 <= direction < np.pi / 4) or (7 * np.pi / 4 <= direction < 2 * np.pi):
                    neighbors = [gradient_magnitude[i, j - 1], gradient_magnitude[i, j + 1]]
                elif (np.pi / 4 <= direction < 3 * np.pi / 4):
                    neighbors = [gradient_magnitude[i - 1, j + 1], gradient_magnitude[i + 1, j - 1]]
                elif (3 * np.pi / 4 <= direction < 5 * np.pi / 4):
                    neighbors = [gradient_magnitude[i - 1, j], gradient_magnitude[i + 1, j]]
                else:
                    neighbors = [gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]]
                if gradient_magnitude[i, j] >= max(neighbors):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
        return suppressed_image

    def threshold(self, threshold):
        self.copyImage = np.where(self.copyImage > threshold, self.copyImage, 0)
    
    def doubleThreshold(self, lowThreshold, highThreshold):
        self.strongEdges = np.where(self.copyImage > highThreshold, self.copyImage, 0)
        self.weakEdges = np.where((self.copyImage <= highThreshold) & (self.copyImage >= lowThreshold), self.copyImage, 0)
       
    
    def hysteresis(self):
        self.copyImage = self.strongEdges.copy()
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if self.weakEdges[i, j] != 0:
                    if np.max(self.copyImage[i-1:i+2, j-1:j+2]) == 255:
                        self.copyImage[i, j] = 255
                    else:
                        self.copyImage[i, j] = 0


    
    def harrisCornerDetection(self, k, threshold, window_size):
        img_gray = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        Ix = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        Ixx = Ix ** 2
        Iyy = Iy ** 2
        Ixy = Ix * Iy

        height, width = img_gray.shape
        offset = window_size // 2
        R = np.zeros_like(img_gray, dtype=np.float32)

        for y in range(offset, height-offset):
            for x in range(offset, width-offset):
                Sxx = np.sum(Ixx[y-offset:y+offset+1, x-offset:x+offset+1])
                Syy = np.sum(Iyy[y-offset:y+offset+1, x-offset:x+offset+1])
                Sxy = np.sum(Ixy[y-offset:y+offset+1, x-offset:x+offset+1])
                det = Sxx*Syy - Sxy ** 2
                trace = Sxx + Syy
                R[y, x] = det - k*(trace**2)

        # Thresholding the corner response function
        corner_points = np.where(R > threshold * R.max())

        # Create a copy of the image to draw circles on
        img_with_corners = self.data.copy()

        # Draw circles on the image at detected corner points
        for y, x in zip(*corner_points):
            cv2.circle(img_with_corners, (x, y), 4, (0, 255, 0), -1)  # -1 for filled circle

        # Return the copy of the image with detected corners
        return img_with_corners
    
    
    def hough_transform(self, edges, threshhold, theta_res, rho_res):
        
        height, width = edges.shape
        img_diagonal = np.ceil(np.sqrt(height ** 2 + width ** 2))
        max_rho = int(np.ceil(img_diagonal / rho_res)) * rho_res

        #define parameters space for rho and theta
        rhos = np.arange(-max_rho, max_rho + 1, rho_res)
        thetas = np.deg2rad(np.arange(-90, 90, theta_res))
        
       # Create am empty hough accumulator
        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

        # Iterate through all edge points in the image
        y_idxs, x_idxs = np.nonzero(edges)
        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            # Vote in the Hough accumulator
            for j in range(len(thetas)):
                rho = x * np.cos(thetas[j]) + y * np.sin(thetas[j])
                rho_idx = int(np.round((rho + max_rho) / rho_res))
                accumulator[rho_idx, j] += 1

        # Apply thresholding
        accumulator[accumulator < threshhold] = 0
        print("AMIGO")
        
        # Find indices of non-zero values in the accumulator
        rho_idxs, theta_idxs = np.nonzero(accumulator)
        rhos_detected = rhos[rho_idxs]
        thetas_detected = thetas[theta_idxs]

        # Combine rho and theta values into a single list
        detected_lines = list(zip(rhos_detected, thetas_detected))

        return detected_lines

       

    def plot_detected_lines(self, lines):
        # Convert the image to RGB
        if len(self.copyImage.shape) == 2:
            print("HOOLAA")
            self.copyImage = cv2.cvtColor(self.copyImage, cv2.COLOR_GRAY2RGB)
        
        # Create an empty image to draw lines on
        line_image = np.zeros_like(self.copyImage)

        # Iterate through detected lines
        if lines is not None:
            for rho, theta in lines:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)

        # Overlay the lines on the original image
        self.copyImage = cv2.addWeighted(self.copyImage, 0.8, line_image, 1, 0)
        
        


    