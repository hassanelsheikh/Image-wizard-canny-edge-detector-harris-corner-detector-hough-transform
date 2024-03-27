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

    def browse_image(self):
        path = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file', '', 'Image files (*.jpg *.png)')[0]
        self.image.read(path)
        #resize the image to fit the window
        self.image.resize(800, 600)
        self.ui.display_image(self.image.data)

    def apply_canny(self):
        self.processor.apply_canny(self.image)
        self.ui.display_image(self.image.data)