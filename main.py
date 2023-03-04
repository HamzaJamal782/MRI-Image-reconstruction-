from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QDialog
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage
from gui import Ui_MainWindow
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import qimage2ndarray
from PIL import Image, ImageEnhance
import cv2
import numpy as np


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # adding items to combobox
        self.ui.comboBox_property.addItems(
            ["Original", "T1", "T2", "SD"])

        # connect ui elements to functions
        self.ui.pushButton_Browse.clicked.connect(self.browse)
        self.ui.comboBox_property.currentIndexChanged.connect(self.combobox)
        self.ui.label_view_img.mousePressEvent = self.getPixel
    
    # create a function to get the pixel value of the image
    def getPixel(self, event):
        #get image from combobox function and convert it to numpy array to get pixel value
        if self.ui.comboBox_property.currentIndex() == 0:
            img = qimage2ndarray.rgb_view(self.image_orignal)
        elif self.ui.comboBox_property.currentIndex() == 1:
            img = qimage2ndarray.rgb_view(self.t1(self.image))
        elif self.ui.comboBox_property.currentIndex() == 2:
            img = qimage2ndarray.rgb_view(self.t2(self.image))
        elif self.ui.comboBox_property.currentIndex() == 3:
            img = qimage2ndarray.rgb_view(self.SD(self.image))
        else:
            pass
        x = event.pos().x()
        y = event.pos().y()
        print(img[x,y])
        


    def browse(self):
        loadImg = QFileDialog.getOpenFileName(self, 'Open file')
        self.image = cv2.imread(loadImg[0], 0)
        self.image_orignal = qimage2ndarray.array2qimage(self.image)
        self.ui.label_view_img.setPixmap(QPixmap(self.image_orignal))

    # combobox function  for selecting image property
    def combobox(self, index):
        if index == 0:
            self.ui.label_view_img.setPixmap(QPixmap(self.image_orignal))
        elif index == 1:
            self.ui.label_view_img.setPixmap(QPixmap(self.t1(self.image)))
        elif index == 2:
            self.ui.label_view_img.setPixmap(QPixmap(self.t2(self.image)))
        elif index == 3:
            self.ui.label_view_img.setPixmap(QPixmap(self.SD(self.image)))
        else:
            pass

    # map range function from brain tissue properties to image pixel values
    def map_range(self, input_value):
        # 2500 is the max value of the property and 255 is the max pixel value
        output_value = input_value * (255 / 300)
        return output_value

    def t1(self, in_image):
        # Define the conditionals and corresponding values
        conditions = [
            in_image == 255,  # white matter
            in_image == 150,  # gray matter
            in_image == 90,   # fat
            in_image == 0     # water
        ]
        values = [self.map_range(500), self.map_range(
            800), self.map_range(250), self.map_range(3000)]

        # Apply the conditionals and assign values in a single step
        shepp_t1 = np.select(conditions, values, default=255).astype(np.uint8)
        print(shepp_t1)

        # Convert image to qimage
        shepp_t1 = qimage2ndarray.array2qimage(shepp_t1)
        return shepp_t1

    def t2(self, in_image):
        # Define the conditionals and corresponding values
        conditions = [
            in_image == 255,  # white matter
            in_image == 150,  # gray matter
            in_image == 90,   # fat
            in_image == 0     # water
        ]
        values = [self.map_range(80), self.map_range(
            100), self.map_range(55), self.map_range(2000)]

        # Apply the conditionals and assign values in a single step
        shepp_t2 = np.select(conditions, values, default=200).astype(np.uint8)
        print(shepp_t2)

        # Convert image to qimage
        shepp_t2 = qimage2ndarray.array2qimage(shepp_t2)

        return shepp_t2

    def SD(self, in_image):
        # Define the conditionals and corresponding values
        conditions = [
            in_image == 255,  # white matter
            in_image == 150,  # gray matter
            in_image == 90,   # fat
            in_image == 0     # water
        ]
        values = [self.map_range(0.1), self.map_range(
            0.2), self.map_range(0.5), self.map_range(0.7)]

        # Apply the conditionals and assign values in a single step
        shepp_SD = np.select(conditions, values, default=0.7).astype(np.uint8)
        print(shepp_SD)

        # Convert image to qimage
        shepp_SD = qimage2ndarray.array2qimage(shepp_SD)

        return shepp_SD


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())
