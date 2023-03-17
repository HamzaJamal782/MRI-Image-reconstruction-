from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QApplication, QMessageBox
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage, QColor, qRgba
from gui import Ui_MainWindow
# from gui_classic import Ui_MainWindow
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
from PIL import Image, ImageEnhance


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        QApplication.processEvents()

        # initializing variables
        self.count = 0
        self.image = np.zeros((16, 16), dtype=np.uint8)

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
            ["16x16", "32x32", "64x64", "128x128", "256x256", "512x512","1024x1024"])

        # connect ui elements to functions
        self.ui.pushButton_Browse.clicked.connect(self.browse)
        self.ui.pushButton_Plot.clicked.connect(self.upload)
        self.ui.comboBox_property.currentIndexChanged.connect(
            self.combobox_select_property)
        self.ui.label_view_img.mousePressEvent = self.getPixel
        self.ui.comboBox_size.currentIndexChanged.connect(
            self.change_size_combo)
        # self.ui.pushButton_Run.clicked.connect(self.RF)
        self.ui.pushButton_Run_Generat_phantom.clicked.connect(
            self.popUpErrorMsg)
        self.ui.slider_contrast.valueChanged.connect(self.adjust_contrast)
        self.ui.slider_brightness.valueChanged.connect(self.adjusted_brightness)
        self.ui.comboBox_property.currentIndexChanged.connect(self.resetSliders)

        # set default value of combobox_size to empty
        self.ui.comboBox_size.setCurrentIndex(-1)

        # set value of row, col and image from size selected from combobox_size
        self.row = str(self.change_size_combo(0))
        self.col = str(self.change_size_combo(1))
        self.img_size = str(self.change_size_combo(2))

    def browse(self):
        try:
            # check if change size combo is empty
            # if self.ui.comboBox_size.currentText() == "":
                # self.popUpErrorMsg("Please select size of image first")
            # else:
            loadImg = QFileDialog.getOpenFileName(self, 'Open file')
            self.image = cv2.imread(loadImg[0], 0)
            self.rgbImage = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
            self.rgbImage = qimage2ndarray.array2qimage(
                self.image)  # convert numpy array to qimage
            # self.rgbImage = self.rgbImage.scaled(self.change_size_combo(0), self.change_size_combo(1))
            self.rgbImage = self.rgbImage.scaled(720 , 576)
            self.ui.label_view_img.setPixmap(QPixmap(self.rgbImage))
        except Exception as e:
            print(e)

    # function to choose size of image from combobox_size
    def change_size_combo(self, flag):
        try:
            resized_imge = None
            size_map = {
                "16x16": (16, 16),
                "32x32": (32, 32),
                "64x64": (64, 64),
                "128x128": (128, 128),
                "256x256": (256, 256),
                "512x512": (512, 512),
                "1024x1024": (1024, 1024),

            }
            size_str = self.ui.comboBox_size.currentText()

            if size_str in size_map:
                resized_img = qimage2ndarray.rgb_view(self.rgbImage)
                resized_img = cv2.resize(resized_img, size_map[size_str])
                resized_imge = qimage2ndarray.array2qimage(resized_img)
                self.ui.label_view_img.setPixmap(
                    QPixmap(resized_imge))

            if flag == 0:
                return resized_img.shape[0]
            elif flag == 1:
                return resized_img.shape[1]
            else:
                return resized_imge

        except Exception as e:
            print(size_str)

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
        for i, phaseAngle in enumerate(self.decimal_range(0, 360, 5.625)):
            Rz_phase = self.Rz(m.radians(phaseAngle))
            for j in range(63):
                self.M[:, j] = np.dot(Rz_phase, self.rgbImage[:, j])
            # self.freqGradient()

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
        print(self.k_space)

    def upload(self):
        self.figure_sequence.clear()
        self.data = json.load(open("data.json", "r"))
        axs = self.figure_sequence.subplots(4, sharex=True)
        self.figure_sequence.suptitle('Sequence')

        for dictionary in self.data:
            if dictionary['name'] == "RF_Pulse":
                self.RF_duration = np.linspace(0, dictionary['RF_Time'], 100)
                self.RF_amplitude = dictionary["RF_amplitude"] * \
                    abs(np.sinc(self.RF_duration))
                axs[0].plot(self.RF_duration, self.RF_amplitude, color='red')
                axs[0].set_ylabel('RF', fontsize=14,
                                  fontweight='bold', rotation=0, labelpad=20)
                axs[0].set_frame_on(False)
                axs[0].xaxis.set_visible(False)
                axs[0].axhline(y=0, color='black')
                axs[0].tick_params(axis='y', colors='red')

            if dictionary['name'] == "Gx_pulse":
                self.Gx_duration = np.linspace(0, dictionary['Gx_Time'], 100)
                self.Gx_amplitude = dictionary["Gx_amplitude"] * \
                    np.where(self.Gx_duration < dictionary["Gx_Time"],  1, 0)
                axs[1].plot(self.Gx_duration, self.Gx_amplitude, color='green')
                axs[1].set_ylabel('Gx', fontsize=14,
                                  fontweight='bold', rotation=0, labelpad=20)
                axs[1].set_frame_on(False)
                axs[1].xaxis.set_visible(False)
                axs[1].axhline(y=0, color='black')
                axs[1].tick_params(axis='y', colors='green')

            if dictionary['name'] == "Gy_Pulse":
                self.Gy_duration = np.linspace(
                    len(self.RF_duration), dictionary['Gy_Time'], 100)
                self.Gy_amplitude = dictionary["Gy_amplitude"] * np.where(
                    self.Gy_duration > dictionary["Gy_Time"],  1, 0)
                axs[2].plot(self.Gy_duration, self.Gy_amplitude, color='blue')
                axs[2].set_ylabel('Gy', fontsize=14,
                                  fontweight='bold', rotation=0, labelpad=20)
                axs[2].set_frame_on(False)
                axs[2].xaxis.set_visible(False)
                axs[2].axhline(y=0, color='black')
                axs[2].tick_params(axis='y', colors='blue')

            if dictionary['name'] == "Readout":
                self.RO_duration = np.linspace(
                    len(self.RF_duration), dictionary['Readout_Time'], 100)
                self.RO_amplitude = abs(np.sinc(self.RO_duration))
                axs[3].plot(self.RO_duration, self.RO_amplitude, color='brown')
                axs[3].set_ylabel('RO', fontsize=14,
                                  fontweight='bold', rotation=0, labelpad=10)
                axs[3].set_frame_on(False)
                axs[3].xaxis.set_visible(True)
                axs[3].axhline(y=0, color='black')
                axs[3].tick_params(axis='y', colors='brown')

        self.canvas_sequence.draw()

    # combobox function  for selecting image property
    def combobox_select_property(self, index):
        try:
            img_copy = self.change_size_combo(2)
            rgb_array = qimage2ndarray.rgb_view(img_copy)

            if index == 0:
                self.ui.label_view_img.setPixmap(QPixmap.fromImage(img_copy))
                return img_copy
            elif index == 1:
                t1_img = self.t1(rgb_array)
                self.ui.label_view_img.setPixmap(QPixmap.fromImage(t1_img))
                return t1_img
            elif index == 2:
                t2_img = self.t2(rgb_array)
                self.ui.label_view_img.setPixmap(QPixmap.fromImage(t2_img))
                return t2_img
            elif index == 3:
                sd_img = self.SD(rgb_array)
                self.ui.label_view_img.setPixmap(QPixmap.fromImage(sd_img))
                return sd_img

        except Exception as e:
            print(e)
            self.popUpErrorMsg("select size first")

    def resetSliders(self):
        # get the current value of the slider
        brughtness_slider_Value = self.ui.slider_brightness.value()
        contrast_slider_Value = self.ui.slider_contrast.value()
        # check if the current value is not equal to the initial value
        if (contrast_slider_Value & brughtness_slider_Value) != 100:
            # set the value of the slider to its initial value
            self.ui.slider_brightness.setValue(100)
            self.ui.slider_contrast.setValue(100)
        
    def adjust_contrast(self, value):
        try:
            # Calculate the contrast factor based on the slider value
            contrast_factor = value /100

            # Create a copy of the original image and apply the contrast adjustment
            index = self.ui.comboBox_property.currentIndex()
            img_1 = self.combobox_select_property(index)

            img_2 = qimage2ndarray.rgb_view(img_1)    
            img_3 = ((img_2/255.0)**(1/contrast_factor))*255.0  

            final_img = qimage2ndarray.array2qimage(img_3)
            
            # Display the adjusted image
            self.ui.label_view_img.setPixmap(QPixmap.fromImage(final_img))
        except Exception as e:
            print(e)
            self.popUpErrorMsg("select image first")
    
    def adjusted_brightness(self, value):
        try:

            # Calculate the contrast factor based on the slider value
            brightness_factor = value /100

            # Create a copy of the original image and apply the contrast adjustment
            index = self.ui.comboBox_property.currentIndex()
            img_1 = self.combobox_select_property(index)

            img_2 = qimage2ndarray.rgb_view(img_1)    

            img_3 = brightness_factor * img_2

            final_img = qimage2ndarray.array2qimage(img_3)
            
            # Display the adjusted image
            self.ui.label_view_img.setPixmap(QPixmap.fromImage(final_img))
        except Exception as e:
            print(e)
            self.popUpErrorMsg("select image first")

    def getPixel(self, event):
        try:

            self.imgT1 = self.t1(qimage2ndarray.rgb_view(self.rgbImage))
            self.imgT2 = self.t2(qimage2ndarray.rgb_view(self.rgbImage))
            self.imgSD = self.SD(qimage2ndarray.rgb_view(self.rgbImage))

            self.imgT1 = qimage2ndarray.rgb_view(self.imgT1)
            self.imgT2 = qimage2ndarray.rgb_view(self.imgT2)
            self.imgSD = qimage2ndarray.rgb_view(self.imgSD)

            # scale the image to the size of the label
            currentWidth = self.ui.label_view_img.width()
            currentHeight = self.ui.label_view_img.height()
            
            # print(type(currentWidth))
            # print(type(int(self.row)))
            # print(type(self.change_size_combo(0)))
        
            x = event.pos().x()
            y = event.pos().y()
            
            # print(round((x*(self.change_size_combo(0)))/currentWidth) , round((y*(self.change_size_combo(1)))/currentHeight))
            new_x = round((x*(self.change_size_combo(0)))/currentWidth)
            new_y = round((y*(self.change_size_combo(1)))/currentHeight)
            
            
            

            '''# Highlight clicked pixel with a dot point
            image_map = {
                1: self.imgT1,
                2: self.imgT2,
                3: self.imgSD
            }

            # get the selected image based on the currentIndex value
            selected_image = image_map.get(self.ui.comboBox_property.currentIndex())

            if selected_image is not None:
                # create a QPainter object for the selected image
                dotted_img = QPainter(selected_image)
                dotted_img.setBrush(QColor(255, 0, 0))
                dotted_img.drawEllipse(x, y, 5, 5)
                dotted_img.end()
                self.ui.label_view_img.setPixmap(QPixmap.fromImage(selected_image))'''
            

            self.ui.lineEdit_t1.setText(str(self.imgT1[new_x,new_y][1]))
            self.ui.lineEdit_t2.setText(str(self.imgT2[new_x,new_y][1]))
            self.ui.lineEdit_sd.setText(str(self.imgSD[new_x,new_y][1]))
        except Exception as e:
            print(e)

    # map range function from brain tissue properties to image pixel values
    def map_range(self, input_value):
        try:
            # 3000 is the max value of the property and 255 is the max pixel value
            output_value = input_value * (255 / 4000) 
            return output_value
        except Exception as e:
            print(e)

    def t1(self, in_image):
        try:
            # Define the conditionals and corresponding values
            conditions = [
                (in_image >= 0) & (in_image < 31),              # Air                       
                (in_image >= 31) & (in_image < 63),             # Fat                       
                (in_image >= 63) & (in_image < 95),             # Blood                     
                (in_image >= 95) & (in_image < 127),            # White matter              
                (in_image >= 127) & (in_image < 159),           # Gray matter               
                (in_image >= 159) & (in_image < 191),           # Muscle                    
                (in_image >= 191) & (in_image < 223),           # Cerebrospinal fluid (CSF) 
                (in_image >= 223) & (in_image <= 225),          # Bone                      
            ]

            
            values = [
                self.map_range(1000),   # Air                           
                self.map_range(300),    # Fat                           
                self.map_range(250),    # Blood                         
                self.map_range(140),    # White matter                  
                self.map_range(120),    # Gray matter                    
                self.map_range(150),    # Muscle                        
                self.map_range(50),     # Cerebrospinal fluid (CSF)     
                self.map_range(10)      # Bone                          
            ]

            # Apply the conditionals and assign values in a single step
            shepp_t1 = np.select(conditions, values,
                                 default=0).astype(np.uint8)

            # Convert image to qimage
            shepp_t1 = qimage2ndarray.array2qimage(shepp_t1)
            return shepp_t1
        except Exception as e:
            print(e)

    def t2(self, in_image):
        try:

            # Define the conditionals and corresponding values
            conditions = [
                (in_image >= 0) & (in_image < 31),              # Air                       
                (in_image >= 31) & (in_image < 63),             # Fat                       
                (in_image >= 63) & (in_image < 95),             # Blood                     
                (in_image >= 95) & (in_image < 127),            # White matter              
                (in_image >= 127) & (in_image < 159),           # Gray matter               
                (in_image >= 159) & (in_image < 191),           # Muscle                    
                (in_image >= 191) & (in_image < 223),           # Cerebrospinal fluid (CSF) 
                (in_image >= 223) & (in_image <= 225),          # Bone                      
            ]

                            
            values = [
                self.map_range(4000),       # Air                           
                self.map_range(250),        # Fat                           
                self.map_range(200),        # Blood                         
                self.map_range(110),        # White matter                  
                self.map_range(100),        # Gray matter                    
                self.map_range(90),         # Muscle                        
                self.map_range(3000),       # Cerebrospinal fluid (CSF)     
                self.map_range(10)          # Bone                          
            ]
            

            # Apply the conditionals and assign values in a single step
            shepp_t2 = np.select(conditions, values,
                                 default=0).astype(np.uint8)
            # print(shepp_t2)

            # Convert image to qimage
            shepp_t2 = qimage2ndarray.array2qimage(shepp_t2)

            return shepp_t2
        except Exception as e:
            print(e)

    def SD(self, in_image):
        try:
            # Define the conditionals and corresponding values
            conditions = [
                (in_image >= 0) & (in_image < 31),              # Air                       
                (in_image >= 31) & (in_image < 63),             # Fat                       
                (in_image >= 63) & (in_image < 95),             # Blood                     
                (in_image >= 95) & (in_image < 127),            # White matter              
                (in_image >= 127) & (in_image < 159),           # Gray matter               
                (in_image >= 159) & (in_image < 191),           # Muscle                    
                (in_image >= 191) & (in_image < 223),           # Cerebrospinal fluid (CSF) 
                (in_image >= 223) & (in_image <= 255),          # Bone                      
            ]

                            
            values = [
                self.map_range(0),          # Air                           
                self.map_range(0),          # Fat                           
                self.map_range(0),          # Blood                         
                self.map_range(0),          # White matter                  
                self.map_range(0),          # Gray matter                    
                self.map_range(0),          # Muscle                        
                self.map_range(0),          # Cerebrospinal fluid (CSF)     
                self.map_range(0)           # Bone                          
            ]
            
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

    # popup msg error function
    def popUpErrorMsg(self, text):
        try:
            msgBox = QMessageBox()
            msgBox.setWindowTitle("Error")
            msgBox.setText(text)
            msgBox.setIcon(msgBox.Critical)
            msgBox.exec_()
        except Exception as e:
            print(e)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())
