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
# plt.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd

# class MplCanvas(FigureCanvasQTAgg):

#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = fig.add_subplot(111)
#         super(MplCanvas, self).__init__(fig)

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.ui.horizontalLayout_4.addWidget(self.canvas)

        # adding items to combobox
        self.ui.comboBox_property.addItems(
            ["Original", "T1", "T2", "SD"])

        # connect ui elements to functions
        self.ui.pushButton_Browse.clicked.connect(self.plot)
        self.ui.comboBox_property.currentIndexChanged.connect(self.combobox)
        self.ui.label_view_img.mousePressEvent = self.getPixel
    
    # create a function to get the pixel value of the image
    def getPixel(self, event):
        #get image from combobox function and convert it to numpy array to get pixel value
        self.orgImg = qimage2ndarray.rgb_view(self.image_orignal)
        self.imgT1 = qimage2ndarray.rgb_view(self.t1(self.image))
        self.imgT2 = qimage2ndarray.rgb_view(self.t2(self.image))
        self.imgSD = qimage2ndarray.rgb_view(self.SD(self.image))



        # if self.ui.comboBox_property.currentIndex() == 0:
        #     self.img = qimage2ndarray.rgb_view(self.image_orignal)
        # elif self.ui.comboBox_property.currentIndex() == 1:
        #     self.img = qimage2ndarray.rgb_view(self.t1(self.image))
        # elif self.ui.comboBox_property.currentIndex() == 2:
        #     self.img = qimage2ndarray.rgb_view(self.t2(self.image))
        # elif self.ui.comboBox_property.currentIndex() == 3:
        #     self.img = qimage2ndarray.rgb_view(self.SD(self.image))
        # else:
        #     pass
        x = event.pos().x()
        y = event.pos().y()
        print(self.imgT1[x,y])
        print(self.imgT2[x,y])
        print(self.imgSD[x,y])
        self.ui.lineEdit_t1.setText(str(self.imgT1[x,y][1]))
        self.ui.lineEdit_t2.setText(str(self.imgT2[x,y][1]))
        self.ui.lineEdit_sd.setText(str(self.imgSD[x,y][1]))

    def browse(self):
        loadImg = QFileDialog.getOpenFileName(self, 'Open file')
        self.image = cv2.imread(loadImg[0], 0)
        self.image_orignal = qimage2ndarray.array2qimage(self.image)
        self.ui.label_view_img.setPixmap(QPixmap(self.image_orignal))
        self.image.resize(256,256)
        # print(self.image.shape)

    # combobox function  for selecting image property
    def plot(self):
        # sc = MplCanvas(self, width=5, height=4, dpi=100)
        # # Create our pandas DataFrame with some simple
        # # data and headers.
        # df = pd.DataFrame([
        #    [0, 4], [5, 13], [2, 20], [3, 25], [4, 10],
        # ], columns=['A', 'B'])

        #   ---------------------------------------


        # t = [1,2,3,4,5]
        # x = [100,100,100,100,100]
        # y = [50,50,50,50,50]
        # z = [70,70,70,70,70]
        # n = [30,30,30,30,30]
        # # # plot the pandas DataFrame, passing in the
        # # # matplotlib Canvas axes.
        # # df.plot(ax=sc.axes)
        # sc.axes.plot(t,x,t,y
        #              ,t,z,t,n)
        # # sc.axes.tick_params(axis=u'both', which=u'both',length=0)
        # sc.axes.xaxis.set_visible(False)
        # sc.axes.yaxis.set_visible(False)

        # self.setCentralWidget(sc)
        # self.show()


            #   ----------------------------------
#         x = [1,2,3,4,5]
#         y = [3,3,3,3,3]
  
# # plot lines
#         plt.plot(x, y, label = "line 1", linestyle="-")
#         plt.plot(y, x, label = "line 2", linestyle="--")
#         plt.plot(x, np.sin(x), label = "curve 1", linestyle="-.")
#         plt.plot(x, np.cos(x), label = "curve 2", linestyle=":")
        
#         self.setCentralWidget(sc)
#         self.show()
        self.figure.clear()
        x = np.linspace(-10, 10, 100)
        y1 = np.sinc(x)
        y2 = np.cos(x)
        y3 = np.tan(x)

        # # Create a figure with 3 subplots
        # # self.axis_zero = self.figure.add_subplot(1,1)
        # # self.axis_one = self.figure.add_subplot(2,1)

        # self.figure, axes = self.figure.subplots(3, 1,sharex=True)

        # # Plot the first subplot
        # axes[0].plot(x+10, y1, label='Sin')
        # axes[0].set_xlabel('x')
        # axes[0].set_ylabel('y')
        # axes[0].legend()
        # axes[0].xaxis.set_visible(False)
        # axes[0].set_frame_on(False)


        # # Plot the second subplot
        # axes[1].plot(x, y2, label='Cos')
        # axes[1].set_xlabel('x')
        # axes[1].set_ylabel('y')
        # axes[1].legend()
        # axes[1].xaxis.set_visible(False)
        # axes[1].set_frame_on(False)


        # # Plot the third subplot
        # axes[2].plot(x, y3, label='Tan')
        # axes[2].set_xlabel('x')
        # axes[2].set_ylabel('y')
        # axes[2].legend()
        # axes[2].set_frame_on(False)


        # # Add a title to the figure
        # self.figure.suptitle('Multiple Graphs')
        # gs = self.figure.add_gridspec(5, hspace=0)

        axs = self.figure.subplots(5,sharex=True)
        self.figure.suptitle('Sharing both axes')


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

        self.canvas.draw()
        # Display the plot
        # self.setCentralWidget(sc)
        # self.show()


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

