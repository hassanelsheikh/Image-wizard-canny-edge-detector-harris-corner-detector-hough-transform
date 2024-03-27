import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui
import cv2

from ui.design import Ui_MainWindow


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_2.clicked.connect(self.browse_image)
        self.ui.pushButton.clicked.connect(self.apply_canny)
        self.show()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.pushButton_2.clicked.connect(self.load_image)
        self.ui.pushButton.clicked.connect(self.apply_canny)

        self.before_scene = QGraphicsScene()
        self.after_scene = QGraphicsScene()

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image, self.before_scene)

    def display_image(self, image, scene):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QtGui.QImage(image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        q_pixmap = QtGui.QPixmap.fromImage(q_image)
        pixmap_item = QGraphicsPixmapItem(q_pixmap)
        scene.addItem(pixmap_item)
        if scene == self.before_scene:
            self.ui.Before.setScene(scene)
        else:
            self.ui.after.setScene(scene)
