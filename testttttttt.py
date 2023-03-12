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


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # for plot sequence
        self.figure_sequence = plt.figure()
        self.canvas_sequence = FigureCanvas(self.figure_sequence)
        self.ui.horizontalLayout_Sequence.addWidget(self.canvas_sequence)

        # for plot kspace
        self.figure_kspace = plt.figure()
        self.canvas_kspace = FigureCanvas(self.figure_kspace)
        self.ui.horizontalLayout_Kspace.addWidget(self.canvas_kspace)

        # for plot phantom generated
        # self.figure_phantom_Generated = plt.figure()
        # self.canvas_phantom_Generated = FigureCanvas(self.figure_phantom_Generated)
        # self.ui.horizontalLayout_phantom_generated.addWidget(self.canvas_phantom_Generated)

        # adding items to combobox
        self.ui.comboBox_property.addItems(
            ["Original", "T1", "T2", "SD"])

        # connect ui elements to functions
        self.ui.pushButton_Browse.clicked.connect(self.browse)
        self.ui.pushButton_Plot.clicked.connect(self.plot)
        # self.ui.pushButton_Run.clicked.connect(self.freqGradient)
        self.ui.comboBox_property.currentIndexChanged.connect(self.combobox)
        self.ui.label_view_img.mousePressEvent = self.getPixel

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

            x = int(((event.pos().x())*64) / currentWidth)
            y = int(((event.pos().y())*64) / currentHeight)

            self.ui.lineEdit_t1.setText(str(self.imgT1[x, y][1]))
            self.ui.lineEdit_t2.setText(str(self.imgT2[x, y][1]))
            self.ui.lineEdit_sd.setText(str(self.imgSD[x, y][1]))
        except Exception as e:
            print(e)

    def browse(self):
        try:
            loadImg = QFileDialog.getOpenFileName(self, 'Open file')
            self.image = cv2.imread(loadImg[0], 0)
            self.image = cv2.resize(self.image, (64, 64))
            # print(self.image.shape)
            self.rgbImage = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
            self.M = self.rgbImage
            self.k_space = self.rgbImage
            self.image_orignal = qimage2ndarray.array2qimage(self.image)
            self.ui.label_view_img.setPixmap(QPixmap(self.image_orignal))
            # print(self.rgbImage)
            print(self.rgbImage[0][0])
            plt.imshow(self.image, cmap='gray')
            self.RF()

        except Exception as e:
            print(e)

    def Rz(self, theta):
        return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                        [m.sin(theta), m.cos(theta), 0],
                        [0, 0, 1]])

    def RF(self):
        self.rgbImage = np.dot(self.Rx(m.radians(90)), self.rgbImage)
        self.phaseGradient()

    def phaseGradient(self):
        for i, phaseAngle in enumerate(self.decimal_range(0, 360, 5.625)):
            Rz_phase = self.Rz(m.radians(phaseAngle))
            for j in range(63):
                self.M[:, j] = np.dot(Rz_phase, self.rgbImage[:, j])
            self.freqGradient()

    def freqGradient(self):
        self.figure_kspace.clear()
        axs =   self.figure_kspace.add_subplot(111)
        axs.set_title("K-Space")
        axs.set_xlabel("Phase")
        axs.set_ylabel("Frequency")
        axs.imshow(self.k_space, cmap='gray')
        
        for i, freqAngle in enumerate(self.decimal_range(0, 360, 5.625)):
            Rz_freq = self.Rz(m.radians(freqAngle))
            PixelSummation = np.zeros(3)
            for j in range(63):
                self.M[:, j] = np.dot(Rz_freq, self.M[:, j])
                PixelSummation += self.M[:, j]
            self.k_space[i, :] = PixelSummation
        
        self.canvas_kspace.draw()
        self.canvas_kspace.flush_events()
        
    def Rx(self,theta):
        return np.matrix([[ 1          , 0           , 0],
                        [ 0, m.cos(theta),-m.sin(theta)],
                        [ 0, m.sin(theta), m.cos(theta)]])

    def decimal_range(start, stop, increment):
        while start < stop: 
            yield start
            start += increment

    def plot(self):
        try:
            self.figure_sequence.clear()
            axs = self.figure_sequence.subplots(5, sharex=True)
            self.figure_sequence.suptitle('Sequence')
            
            x = np.linspace(-10, 10, 500)
            y1 = np.sinc(x)
            y2 = np.cos(x)
            y3 = np.tan(x)

          
           
            axs[0].plot(x, y1 ** 2)
            axs[0].set_ylabel('RF')
            axs[0].set_frame_on(False)
            axs[0].xaxis.set_visible(False)

            axs[1].plot(x, 0.3 * y2)
            axs[1].set_ylabel('GX')
            axs[1].set_frame_on(False)
            axs[1].xaxis.set_visible(False)

            axs[2].plot(x, 0.3 * y2)
            axs[2].set_ylabel('GY')
            axs[2].xaxis.set_visible(False)
            axs[2].set_frame_on(False)

            axs[3].plot(x, 0.3 * y2)
            axs[3].set_ylabel('GZ')
            axs[3].xaxis.set_visible(False)
            axs[3].set_frame_on(False)

            axs[4].plot(x, y3)
            axs[4].set_ylabel('Read Out')
            axs[4].set_frame_on(False)

            self.canvas_sequence.draw()
        except Exception as e:
            print(e)

    # combobox function  for selecting image property
    def combobox(self, index):
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
            in_image = cv2.resize(in_image, (256, 256))
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

            # Convert image to qimage
            shepp_t1 = qimage2ndarray.array2qimage(shepp_t1)
            return shepp_t1
        except Exception as e:
            print(e)

    def t2(self, in_image):
        try:
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

            # Convert image to qimage
            shepp_t2 = qimage2ndarray.array2qimage(shepp_t2)

            return shepp_t2
        except Exception as e:
            print(e)

    def SD(self, in_image):
        try:
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

            # Convert image to qimage
            shepp_SD = qimage2ndarray.array2qimage(shepp_SD)

            return shepp_SD
        except Exception as e:
            print(e)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())
