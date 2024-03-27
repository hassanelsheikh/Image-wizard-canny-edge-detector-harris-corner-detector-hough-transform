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

       # Connect input fields to slots
        self.ui.kernelLineEdit.textChanged.connect(self.update_kernel)
        self.ui.lowThresholdLineEdit.textChanged.connect(self.update_low_threshold)
        self.ui.highThresholdLineEdit.textChanged.connect(self.update_high_threshold)      

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

    def apply_canny(self):
        # Reset the copy image
        self.image.copyImage = self.image.data.copy()
        try:
            self.image.copyImage = self.processor.apply_canny(self.image, self.kernel_size, self.low_threshold, self.high_threshold)
            print("Canny edge detector applied")
            self.ui.display_result_image(self.image.copyImage)
        except AttributeError as e:
            print("Error: ", e)
            return