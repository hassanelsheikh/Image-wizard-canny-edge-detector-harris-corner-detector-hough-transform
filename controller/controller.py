from model.image_processor import imageProcessor
from model.image_ import Image
from view.main_window import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication
import sys
import cv2
import numpy as np


class Controller:
    def __init__(self, ui):
        self.ui = ui
        self.image = Image(None, 0, 0)
        self.processor = imageProcessor()
        
        # Connect signals to slots
        self.ui.pushButton.clicked.connect(self.apply_canny)
        self.ui.pushButton_2.clicked.connect(self.browse_image)
        self.ui.pushButton_4.clicked.connect(self.apply_harris_transform)

       # Connect input fields to slots
        self.ui.kernelLineEdit.textChanged.connect(self.update_kernel)
        self.ui.lowThresholdLineEdit.textChanged.connect(self.update_low_threshold)
        self.ui.highThresholdLineEdit.textChanged.connect(self.update_high_threshold) 
        self.ui.sigmaLineEdit.textChanged.connect(self.update_sigma)     

        # Connect Hough line detection parameters to slots
        self.ui.rhosLineEdit.textChanged.connect(self.update_rho_res)
        self.ui.thetasLineEdit.textChanged.connect(self.update_theta_res)
        self.ui.thresholdRatioLineEdit.textChanged.connect(self.update_threshold_ratio)
        self.ui.pushButton_3.clicked.connect(self.apply_hough_transform)

        # Connect Harris corner detection parameters to slots
        self.ui.windowSizeLineEdit.textChanged.connect(self.update_window_size)
        self.ui.k_valueLineEdit.textChanged.connect(self.update_k)
        self.ui.thresholdLineEdit.textChanged.connect(self.update_threshold)

        

    def browse_image(self):
        path = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '', 'Image files (*.jpg *.png)')[0]
        self.image.read(path)
        # Resize the image to fit the window
        self.image.resize(800, 600) 
        self.ui.display_initial_image(self.image.data)

    def update_kernel(self, text):
        try:
            self.kernel_size = int(text)
        except ValueError as e:
            print("Error: ", e)
            return
        
    def update_sigma(self, text):
        try:
            self.sigma = float(text)
        except ValueError as e:
            print("Error: ", e)
            return


    def update_low_threshold(self, text):
        try: 
            self.low_threshold = int(text)
        except ValueError as e:
            print("Error: ", e)
            return

    def update_high_threshold(self, text):
        try:
            self.high_threshold = int(text)
        except ValueError as e:
            print("Error: ", e)
            return
        
    def update_theta_res(self, text):
        try:
            self.theta_res = float(text)
        except ValueError as e:
            print("Error: ", e)
            return
        
    def update_rho_res(self, text):
        try:
            self.rho_res = float(text)
        except ValueError as e:
            print("Error: ", e)
            return
    
    def update_threshold_ratio(self, text):
        try:
            self.threshold_ratio = float(text)
        except ValueError as e:
            print("Error: ", e)
            return
        
    def update_window_size(self, text):
        try:
            self.window_size = int(text)
        except ValueError as e:
            print("Error: ", e)
            return
        
    def update_k(self, text):
        try:
            self.k = float(text)
        except ValueError as e:
            print("Error: ", e)
            return
        
    def update_threshold(self, text):
        try:
            self.threshold = float(text)
        except ValueError as e:
            print("Error: ", e)
            return
    

    def apply_canny(self):
        # Reset the copy image
        self.image.copyImage = self.image.data.copy()
        try:
            self.image.copyImage = self.processor.apply_canny(self.image, self.kernel_size, self.sigma, self.low_threshold, self.high_threshold)
            print("Canny edge detector applied")
            self.ui.display_result_image(self.image.copyImage, "gray")
        except AttributeError as e:
            print("Error: ", e)
            return
        
    def apply_harris_transform(self):
        # Reset the copy image
        self.image.copyImage = self.image.data.copy()
        try:
            self.image.copyImage = self.processor.apply_harris_transform(self.image, self.window_size, self.k, self.threshold)
            print("Harris corner detector applied")
            self.ui.display_result_image(self.image.copyImage, "rgb")
        except AttributeError as e:
            print("Error: ", e)
            return
        
    def apply_hough_transform(self):
        # Reset the copy image
        self.image.copyImage = self.image.data.copy()
        try:
            self.image.copyImage = self.processor.apply_hough_transform(self.image, self.threshold_ratio, self.theta_res, self.rho_res)
            print("Hough line detector applied")
            self.ui.display_result_image(self.image.copyImage, "rgb")
        except AttributeError as e:
            print("Error: ", e)
            return
        
