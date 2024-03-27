import cv2
import numpy as np

class Image:
    def __init__(self, data, width, height):
        self.data = data
        self.width = width
        self.height = height
        self.strongEdges = None
        self.weakEdges = None

    def read(self, path):
        self.data = cv2.imread(path)
        self.width = self.data.shape[1]
        self.height = self.data.shape[0]

    def resize(self, width, height):
        self.data = cv2.resize(self.data, (width, height))
        self.width = width
        self.height = height

    def convertToGray(self):
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)

    def gaussianBlur(self, kernel_size):
        #Generate Gaussian kernel
        kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size**2
        kernel /= kernel_size**2

        #convolve the image with the kernel
        self.image.data = cv2.filter2D(self.image.data, -1, kernel)

    def gradientIntensity(self):
        #soebel kernels
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        #convolve the image with the kernels
        gradient_x = cv2.filter2D(self.image.data, -1, kernel_x)
        gradient_y = cv2.filter2D(self.image.data, -1, kernel_y)

        #compute the gradient intensity
        gradient_intensity = np.sqrt(gradient_x**2 + gradient_y**2)

        #normalize the gradient intensity
        gradient_intensity = (gradient_intensity / gradient_intensity.max() * 255).astype(np.uint8) 

        self.image.data = gradient_intensity

    def threshold(self, threshold):
        self.image.data = np.where(self.image.data > threshold, self.image.data, 0)

    
    def doubleThreshold(self, lowThreshold, highThreshold):
        self.strongEdges = np.where(self.image.data > highThreshold, self.image.data, 0)
        self.weakEdges = np.where((self.image.data <= highThreshold) & (self.image.data >= lowThreshold), self.image.data, 0)
       
    
    def hysteresis(self):
        self.image.data = self.strongEdges.copy()
        for i in range(1, self.height-1):
            for j in range(1, self.width-1):
                if self.weakEdges[i, j] != 0:
                    if np.max(self.image.data[i-1:i+2, j-1:j+2]) == 255:
                        self.image.data[i, j] = 255
                    else:
                        self.image.data[i, j] = 0


    