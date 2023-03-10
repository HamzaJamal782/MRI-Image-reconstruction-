from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QDialog
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage
from gui import Ui_MainWindow
import sys
import math as m
import numpy as np
import matplotlib.pyplot as plt
import cv2
import qimage2ndarray
from PIL import Image, ImageEnhance
import cv2
import numpy as np
# plt.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.patches as patches
import json


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # initializing variables
        self.image = np.zeros((16, 16), dtype=np.uint8)
        self.count = 0

        # for plot sequence
        self.figure_sequence = plt.figure()
        self.canvas_sequence = FigureCanvas(self.figure_sequence)
        self.ui.horizontalLayout_Sequence.addWidget(self.canvas_sequence)

        # for plot kspace
        self.figure_kspace = plt.figure()
        self.canvas_kspace = FigureCanvas(self.figure_kspace)
        self.ui.horizontalLayout_Kspace.addWidget(self.canvas_kspace)

        # for plot phantom
        self.figure_phantom = plt.figure()
        self.canvas_phantom = FigureCanvas(self.figure_phantom)
        self.ui.horizontalLayout_phantom.addWidget(self.canvas_phantom)

        # adding items to combobox properties
        self.ui.comboBox_property.addItems(
            ["Original", "T1", "T2", "SD"])

        # adding items to combobox size
        self.ui.comboBox_size.addItems(
            ["16x16", "32x32", "64x64", "128x128", "256x256"])

        # connect ui elements to functions
        self.ui.pushButton_Browse.clicked.connect(self.browse)
        self.ui.pushButton_Plot.clicked.connect(self.upload)
        self.ui.comboBox_property.currentIndexChanged.connect(
            self.combobox_select_property)
        self.ui.label_view_img.mousePressEvent = self.getPixel
        self.ui.comboBox_size.currentIndexChanged.connect(
            self.change_size_combo)

        # set default value of combobox_size to empty
        self.ui.comboBox_size.setCurrentIndex(-1)

        # set value of row and col from size selected from combobox_size
        self.row = str(self.change_size_combo(0))
        self.col = str(self.change_size_combo(0))

    # function to choose size of image from combobox_size

    def change_size_combo(self, flag):
        try:
            size_map = {
                "16x16": (16, 16),
                "32x32": (32, 32),
                "64x64": (64, 64),
                "128x128": (128, 128),
                "256x256": (256, 256)
            }
            size_str = self.ui.comboBox_size.currentText()
            if size_str in size_map:
                self.image = cv2.resize(self.image, size_map[size_str])

            if flag == 0:
                return self.image.shape[0]
            else:
                return self.image.shape[:2]

        except Exception as e:
            print(e)

    # create a function to get the pixel value of the image
    def getPixel(self, event):
        try:
            # get image from combobox function and convert it to numpy array to get pixel value
            self.orgImg = qimage2ndarray.rgb_view(self.image_orignal)
            self.imgT1 = qimage2ndarray.rgb_view(self.t1(self.image))
            self.imgT2 = qimage2ndarray.rgb_view(self.t2(self.image))
            self.imgSD = qimage2ndarray.rgb_view(self.SD(self.image))

            currentWidth = self.ui.label_view_img.width()
            currentHeight = self.ui.label_view_img.height()

            x = int(((event.pos().x())*10) / currentWidth)
            y = int(((event.pos().y())*10) / currentHeight)

            self.ui.lineEdit_t1.setText(str(self.imgT1[x, y][1]))
            self.ui.lineEdit_t2.setText(str(self.imgT2[x, y][1]))
            self.ui.lineEdit_sd.setText(str(self.imgSD[x, y][1]))
        except Exception as e:
            print(e)

    def browse(self):
        # try:
        loadImg = QFileDialog.getOpenFileName(self, 'Open file')
        self.image = cv2.imread(loadImg[0], 0)
        # self.image = cv2.resize(self.image, (10, 10))
        self.rgbImage = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        self.rgbImage = cv2.resize(self.image, (int(self.row), int(self.col)))
        self.image_orignal = qimage2ndarray.array2qimage(self.image)
        self.ui.label_view_img.setPixmap(QPixmap(self.image_orignal))
        # print(self.rgbImage)
        # print(self.rgbImage[0][5])
        plt.imshow(self.image, cmap='gray')
        self.RF()
        # except Exception as e:
        #     print(e)

    def Rx(self, theta):
        return np.matrix([[1, 0, 0],
                          [0, m.cos(theta), -m.sin(theta)],
                          [0, m.sin(theta), m.cos(theta)]])

    def Ry(self, theta):
        return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                          [0, 1, 0],
                          [-m.sin(theta), 0, m.cos(theta)]])

    def Rz(self, theta):
        return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                          [m.sin(theta), m.cos(theta), 0],
                          [0, 0, 1]])

    def RF(self):
        self.M = [[0 for i in np.arange(int(self.row))]
                  for j in np.arange(int(self.col))]
        self.k_space = [[0 for i in np.arange(
            int(self.row))] for j in np.arange(int(self.col))]
        for i in range(int(self.row)):
            for j in range(int(self.col)):
                self.M[i][j] = np.ravel(
                    np.dot(self.Rx(m.radians(90)), self.rgbImage[i][j]))

        self.phaseGradient()

    def phaseGradient(self):
        self.i1 = -1
        self.phaseValue = 360 / int(self.row)
        for phaseAngle in np.arange(0, 360, self.phaseValue):
            self.i1 = self.i1 + 1
            for i in np.arange(int(self.row)):
                for j in np.arange(int(self.col)):
                    self.M[i][j] = np.ravel(
                        np.dot(self.M[i][j], self.Rz(m.radians(phaseAngle))))  # M1

            self.freqGradient()

    def freqGradient(self):
        self.i2 = -1
        for freqAngle in np.arange(0, 360, self.phaseValue):
            self.i2 = self.i2 + 1
            PixelSummation = [0, 0, 0]
            for i in np.arange(int(self.row)):
                for j in np.arange(int(self.col)):
                    self.M[i][j] = np.ravel(
                        np.dot(self.M[i][j], self.Rz(m.radians(freqAngle))))
                    PixelSummation = PixelSummation + self.M[i][j]

            self.k_space[self.i1][self.i2] = PixelSummation
        # print(self.k_space)

    def upload(self):
        f = open("../Image-reconstruction--main/data.json", "r")
        self.data = json.load(f)
        for i in self.data:
            print(i)

        self.plot()

    def plot(self):
        try:
            self.figure_sequence.clear()
            x1 = np.linspace(-50, 100, 500)
            y1 = np.sinc(x1)
            x2 = np.linspace(-50, 100, 6)
            y2 = [0, 0, 1.5, 1.5, 0, 0]
            y3 = np.tan(x1)

            axs = self.figure_sequence.subplots(5)
            self.figure_sequence.suptitle('Sequence')

            axs[0].plot(x1, y1)
            axs[0].set_ylabel('RF')
            axs[0].set_frame_on(False)
            axs[0].xaxis.set_visible(False)

            axs[1].plot(x2, y2)
            axs[1].set_ylabel('GX')
            axs[1].set_frame_on(False)
            axs[1].xaxis.set_visible(False)

            axs[2].plot(x2, y2)
            axs[2].set_ylabel('GY')
            axs[2].xaxis.set_visible(False)
            axs[2].set_frame_on(False)

            axs[3].plot(x2, y2)
            axs[3].set_ylabel('GZ')
            axs[3].xaxis.set_visible(False)
            axs[3].set_frame_on(False)

            axs[4].plot(x1, y3)
            axs[4].set_ylabel('Read Out')
            axs[4].set_frame_on(False)

            self.canvas_sequence.draw()
        except Exception as e:
            print(e)

    # combobox function  for selecting image property
    def combobox_select_property(self, index):
        try:
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
        except Exception as e:
            print(e)

    # map range function from brain tissue properties to image pixel values
    def map_range(self, input_value):
        try:
            # 2500 is the max value of the property and 255 is the max pixel value
            output_value = input_value * (255 / 300)
            return output_value
        except Exception as e:
            print(e)

    def t1(self, in_image):
        try:
            in_image = cv2.resize(in_image, self.change_size_combo(1))
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
            shepp_t1 = np.select(conditions, values,
                                 default=255).astype(np.uint8)
            # print(shepp_t1)

            # Convert image to qimage
            shepp_t1 = qimage2ndarray.array2qimage(shepp_t1)
            return shepp_t1
        except Exception as e:
            print(e)

    def t2(self, in_image):
        try:
            in_image = cv2.resize(in_image, self.change_size_combo(1))
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
            shepp_t2 = np.select(conditions, values,
                                 default=200).astype(np.uint8)
            # print(shepp_t2)

            # Convert image to qimage
            shepp_t2 = qimage2ndarray.array2qimage(shepp_t2)

            return shepp_t2
        except Exception as e:
            print(e)

    def SD(self, in_image):
        try:
            in_image = cv2.resize(in_image, self.change_size_combo(1))
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
            shepp_SD = np.select(conditions, values,
                                 default=0.7).astype(np.uint8)
            # print(shepp_SD)

            # Convert image to qimage
            shepp_SD = qimage2ndarray.array2qimage(shepp_SD)

            return shepp_SD
        except Exception as e:
            print(e)

    def decimal_range(start, stop, increment):
        while start < stop:  # and not math.isclose(start, stop): Py>3.5
            yield start
            start += increment


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())
