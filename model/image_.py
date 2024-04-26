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

    def gaussianBlur(self, kernel_size, sigma):
        # Generate Gaussian kernel
        kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2
        kernel /= kernel_size**2
        self.copyImage = cv2.GaussianBlur(self.copyImage, (kernel_size, kernel_size), sigma)
        

        # Convolve the image with the kernel
        self.copyImage = cv2.filter2D(self.copyImage, -1, kernel)

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


    def hough_transform(image, theta_res=1, rho_res=1):
        height, width = image.shape
        max_rho = int(np.sqrt(height*2 + width*2))
        rhos = np.arange(-max_rho, max_rho, rho_res)
        thetas = np.deg2rad(np.arange(-90, 90, theta_res))

        accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

        y_idxs, x_idxs = np.nonzero(image)  # Get coordinates of non-zero values (edge points)

        for i in range(len(x_idxs)):
            x = x_idxs[i]
            y = y_idxs[i]

            for j in range(len(thetas)):
                rho = int((x * np.cos(thetas[j]) + y * np.sin(thetas[j])) / rho_res) + max_rho
                accumulator[rho, j] += 1

        return accumulator, thetas, rhos

    def detect_lines(accumulator, thetas, rhos, threshold):
        lines = []
        for rho_idx in range(accumulator.shape[0]):
            for theta_idx in range(accumulator.shape[1]):
                if accumulator[rho_idx, theta_idx] > threshold:
                    rho = rhos[rho_idx]
                    theta = thetas[theta_idx]
                    lines.append((rho, theta))
        return lines



    def plot_detected_lines(image, accumulator, rhos, thetas, threshold_ratio=0.5):
        # Set a dynamic threshold based on the max value in the accumulator
        threshold = threshold_ratio * accumulator.max()

        lines = detect_lines(accumulator, thetas, rhos, threshold)

        # Create a copy of the image to draw lines on
        line_image = np.copy(image)

        print(f"Threshold: {threshold}")
        print("Detected lines (rho, theta):")

        # Plot the lines on the copy of the image
        for rho, theta in lines:
            print(f"({rho}, {theta})")  # Debug print statement
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            print(f"Line endpoints: ({x1}, {y1}), ({x2}, {y2})")  # Debug print statement

            # Draw lines on the image copy
            plt.plot((x1, x2), (y1, y2), 'red', linewidth=2)
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        axes[1].imshow(line_image, cmap='gray')
        axes[1].set_title('Detected Lines')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
                        
        


    