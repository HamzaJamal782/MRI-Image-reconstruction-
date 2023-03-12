from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog,QApplication,QMessageBox
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
import cmath

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        QApplication.processEvents()

        #error message
        self.msg = QMessageBox()
        self.msg.setWindowTitle("Warning")
        self.msg.setIcon(QMessageBox.Critical)

        # initializing variables
        # self.image = np.zeros((16, 16), dtype=np.uint8)
        self.count = 0
        self.flag = 0

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
        
        # set default value of combobox_size to empty
        self.ui.comboBox_size.setCurrentIndex(-1)

        # connect ui elements to functions
        self.ui.pushButton_Browse.clicked.connect(self.browse)
        self.ui.pushButton_Plot.clicked.connect(self.plot)
        self.ui.pushButton_Run.clicked.connect(self.RF)
        self.ui.comboBox_property.currentIndexChanged.connect(
            self.combobox_select_property)
        self.ui.label_view_img.mousePressEvent = self.getPixel
        self.ui.comboBox_size.currentIndexChanged.connect(self.change_size_combo)

        

        # set value of row and col from size selected from combobox_size
        # self.row = str(self.change_size_combo(0))
        # self.col = str(self.change_size_combo(0))

    # function to choose size of image from combobox_size

    def change_size_combo(self):
        try:
            size_map = {
                "16x16": 16,
                "32x32": 32,
                "64x64": 64,
                "128x128": 128,
                "256x256": 256
            }
            size_str = self.ui.comboBox_size.currentText()
            
            self.row = size_map[size_str]
            self.col = size_map[size_str]
            print(self.flag)
            if self.flag == 1:
                self.plot_phantom()

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
        try:
            if self.ui.comboBox_size.currentIndex() == -1:
                self.msg.setText("Please choose a size first!")
                self.msg.exec_()
            else:
                self.flag = 1
                self.loadImg = QFileDialog.getOpenFileName(self, 'Open file')
                self.plot_phantom()

                # self.RF()
        except Exception as e:
            print(e)

    def plot_phantom(self):
        self.image = cv2.imread(self.loadImg[0], 0)
        self.image = cv2.resize(self.image, (int(self.row), int(self.col)))
        print("111")
        print(self.image.shape)
        self.rgbImage = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        # self.rgbImage = cv2.resize(self.image, (int(self.row), int(self.col)))
        self.image_orignal = qimage2ndarray.array2qimage(self.image)
        # self.image_orignal = self.image_orignal.scaled(int(self.row), int(self.col))
        self.ui.label_view_img.setPixmap(QPixmap(self.image_orignal))
        plt.imshow(self.image, cmap='gray')

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
        self.M1 = [[0 for i in np.arange(int(self.row))]
                  for j in np.arange(int(self.col))]
        self.M2 = [[0 for i in np.arange(int(self.row))]
                  for j in np.arange(int(self.col))]
        self.k_space = [[0 for i in np.arange(int(self.row))] for j in np.arange(int(self.col))]
        self.fourier = [[0 for i in np.arange(int(self.row))] for j in np.arange(int(self.col))]
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
                    self.M1[i][j] = np.ravel(
                        np.dot(self.M[i][j], self.Rz(m.radians(phaseAngle))))  # M1

            self.freqGradient()

    def freqGradient(self):
        self.i2 = -1
        for freqAngle in np.arange(0, 360, self.phaseValue):
            self.i2 = self.i2 + 1
            PixelSummation = [0, 0, 0]
            for i in np.arange(int(self.row)):
                for j in np.arange(int(self.col)):
                    self.M2[i][j] = np.ravel(np.dot(self.M1[i][j],self.Rz(m.radians(freqAngle))))   #M
                    PixelSummation = PixelSummation + self.M2[i][j]
                    # PixelSummation = PixelSummation + self.M[i][j]
            
            # print(type(PixelSummation))
            # self.ui.lineEdit_t1.setText(str(PixelSummation))
            # time.sleep(1)
            # self.canvas_kspace = PixelSummation[0]
            # self.canvas_kspace.draw()
            self.k_space[self.i1][self.i2] = PixelSummation
            self.fourier[self.i1][self.i2] = complex(PixelSummation[0] , PixelSummation[1])
            print(self.fourier[self.i1][self.i2])
            self.canvas_kspace = self.fourier
            self.canvas_kspace.draw()


        

        # print(self.k_space)
    # def generat_fourier(self):
    #     self.fourier = [[0 for i in np.arange(int(self.row))] for j in np.arange(int(self.col))]
    #     for ii in self.k_space:
    #         ii[0]
    #         ii[1]

    def upload(self):
        f = open("../Image-reconstruction--main/data.json", "r")
        self.data = json.load(f)
        for i in self.data:
            print(i)

        self.plot()

    def plot(self):
        try:
            self.figure_sequence.clear()
            t = np.linspace(0, 50, 1000)
            y1 = np.sinc(t)
            y2 = np.where(np.sin(t) > 0, 1, -1)
            y3 = np.sin(t)

            axs = self.figure_sequence.subplots(5)
            self.figure_sequence.suptitle('Sequence')

            axs[0].plot(t, y1 , color='red')
            axs[0].set_ylabel('RF')
            axs[0].set_frame_on(False)
            axs[0].xaxis.set_visible(False)
            axs[0].axhline(y=0, color='black')
            axs[0].tick_params(axis='y', colors= 'red')

            axs[1].plot(t, y2, color='green')
            axs[1].set_ylabel('GX')
            axs[1].set_frame_on(False)
            axs[1].xaxis.set_visible(False)
            axs[1].axhline(y=0, color='black')
            axs[1].tick_params(axis='y', colors='green')

            axs[2].plot(t, y2,    color='blue')
            axs[2].set_ylabel('GY')
            axs[2].xaxis.set_visible(False)
            axs[2].set_frame_on(False)
            axs[2].axhline(y=0, color='black')
            axs[2].tick_params(axis='y', colors='blue')

            axs[3].plot(t, y2, color='black')
            axs[3].set_ylabel('GZ')
            axs[3].xaxis.set_visible(False)
            axs[3].set_frame_on(False)
            axs[3].axhline(y=0, color='black')
            axs[3].tick_params(axis='y', colors='brown')

            axs[4].plot(t, y3)
            axs[4].set_ylabel('Read Out')
            axs[4].set_frame_on(False)
            axs[4].axhline(y=0, color='black')
            

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
            in_image = cv2.resize(in_image, (self.row,self.col))
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
            in_image = cv2.resize(in_image, (self.row,self.col))
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
            in_image = cv2.resize(in_image, (self.row,self.col))
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
