from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog,QApplication,QMessageBox
from PyQt5.QtGui import QPixmap, QPainter, QColor,qRgba
from PyQt5.QtGui import QImage,QPen
from gui_classic import Ui_MainWindow
import sys
import math as m
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Rectangle
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
        # self.image = np.zeros((16, 16,3))
        # self.image2 = np.matrix(self.image)

        self.count = 0
        self.flag = 0
        self.phantom_flag = False

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
        self.ui.pushButton_Plot.clicked.connect(self.upload)
        self.ui.pushButton_Run.clicked.connect(self.RF)
        self.ui.pushButton_Run_Generat_phantom.clicked.connect(self.generat_phantom)
        self.ui.comboBox_property.currentIndexChanged.connect(
            self.combobox_select_property)
        self.ui.label_view_img.mousePressEvent = self.getPixel
        # self.ui.label_view_img.mousePressEvent = self.Highlighte()

        self.ui.comboBox_size.currentIndexChanged.connect(self.change_size_combo)
        self.ui.comboBox_size.currentIndexChanged.connect(self.clearFigures)

        self.ui.slider_contrast.valueChanged.connect(self.adjust_contrast)
        self.ui.slider_brightness.valueChanged.connect(self.adjusted_brightness)
        self.ui.comboBox_property.currentIndexChanged.connect(self.resetSliders)
        # self.ui.pushButton_Run_Generat_phantom.clicked.connect(
        #     self.popUpErrorMsg)


        

        # set value of row and col from size selected from combobox_size
        # self.row = str(self.change_size_combo(0))
        # self.col = str(self.change_size_combo(0))

    # function to choose size of image from combobox_size

    def change_size_combo(self):
        try:
            
            self.resetSliders()
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

    def clearFigures(self):
        if self.ui.comboBox_property.currentIndex != 0:
            self.ui.comboBox_property.setCurrentIndex(0)

        self.figure_kspace.clear()
        self.canvas_kspace.draw()

        self.figure_phantom.clear()
        self.canvas_phantom.draw()

    def Highlighte(self):
        image_map = {
                0: self.orgImg,
                1: self.imgT1,
                2: self.imgT2,
                3: self.imgSD
            }

            # get the selected image based on the currentIndex value
        selected_image = image_map.get(self.ui.comboBox_property.currentIndex())
        
        #convert the selected image to a QImage
        selected_image = qimage2ndarray.array2qimage(selected_image)
        
            
        # print(type(selected_image))

        if selected_image is not None:
            # create a QPainter object for the selected image
            dotted_img = QPainter(selected_image)
            # dotted_img = QPainter(self.pixmap())
            pen = QPen(Qt.red, 1)
            dotted_img.setPen(pen)
            print(self.ui.label_view_img.width()/self.row)
            print(self.ui.label_view_img.height()/self.row)

            dotted_img.drawRect(self.x, self.y, 1, 1)
            dotted_img.end()
            self.update()
            # Rectangle1 = Rectangle((self.x,self.y),1,1)
            # Rectangle1.set_color(255,0,0)
            # Rectangle1.fill(False)
            # dotted_img.add_patch(Rectangle1)
            
            # dotted_img.end()
            # dotted_img.drawPixmap(patch['x'], patch['y'], patch['pixmap'])
            # dotted_img.setBrush(QColor(255, 0, 0))
            # dotted_img.drawRect(self.rect_start.x(), self.rect_start.y(),
            #             self.rect_end.x() - self.rect_start.x(),
            #             self.rect_end.y() - self.rect_start.y())
            # dotted_img.end()
            self.ui.label_view_img.setPixmap(QPixmap.fromImage(selected_image))
        
    # create a function to get the pixel value of the image
    def getPixel(self, event):
        # try:
            # get image from combobox function and convert it to numpy array to get pixel value
            self.orgImg = qimage2ndarray.rgb_view(self.image_orignal)
            self.imgT1 = qimage2ndarray.rgb_view(self.t1(self.image))
            # print(self.imgT1)
            self.imgT2 = qimage2ndarray.rgb_view(self.t2(self.image))
            # print(self.imgT2)

            self.imgSD = qimage2ndarray.rgb_view(self.SD(self.image))
            # print(self.imgSD)

            currentWidth = self.ui.label_view_img.width()
            currentHeight = self.ui.label_view_img.height()

            self.x = int(((event.pos().x())*self.row) / currentWidth)
            print(self.x)
            self.y = int(((event.pos().y())*self.col) / currentHeight)
            print(self.y)

            self.Highlighte()
            self.ui.lineEdit_t1.setText(str(self.imgT1[self.x, self.y][1]))
            self.ui.lineEdit_t2.setText(str(self.imgT2[self.x, self.y][1]))
            self.ui.lineEdit_sd.setText(str(self.imgSD[self.x, self.y][1]))
        # except Exception as e:
        #     print(e)

    def browse(self):
        # try:
            if self.ui.comboBox_size.currentIndex() == -1:
                self.popUpErrorMsg("select image first")
            else:
                self.flag = 1
                self.loadImg = QFileDialog.getOpenFileName(self, 'Open file')
                self.plot_phantom()
                # self.contrast()

                # self.RF()
        # except Exception as e:
        #     print(e)

    def plot_phantom(self):
        self.image = cv2.imread(self.loadImg[0], 0)
        self.image = cv2.resize(self.image, (self.row, self.col))
        # dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(self.image))
        # print(dark_image_grey_fourier)
        # self.image_converted = np.fft.fftshift(np.fft.fft2(self.image))
        # print(self.image)
        self.rgbImage = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        # dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(self.rgbImage))
        # print(dark_image_grey_fourier)
        
        

        print(self.rgbImage)
        print("111")
        # self.rgbImage = cv2.resize(self.image, (self.row, self.col))
        self.image_orignal = qimage2ndarray.array2qimage(self.image)
        # self.image_orignal = self.image
        # self.image_orignal = self.image_orignal.scaled(int(self.row), int(self.col))
        self.ui.label_view_img.setPixmap(QPixmap(self.image_orignal))
        plt.imshow(self.image, cmap='gray')

    def Rx(self,theta):
        return np.matrix([[ 1, 0           , 0           ],
                        [ 0, m.cos(theta),-m.sin(theta)],
                        [ 0, m.sin(theta), m.cos(theta)]])
 
    def Ry(self,theta):
        return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                        [ 0           , 1, 0           ],
                        [-m.sin(theta), 0, m.cos(theta)]])
    
    def Rz(self,theta):
        return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                        [ m.sin(theta), m.cos(theta) , 0 ],
                        [ 0           , 0            , 1 ]])

    def RF(self):
        for ii in range(self.row):
            for jj in range(self.col):
                self.rgbImage[ii][jj][0]=0
                self.rgbImage[ii][jj][1]=0
        self.M = np.zeros((self.row,self.col,3))
        self.M1 = np.zeros((self.row,self.col,3))
        self.M2 = [[0 for i in np.arange(self.row)]
                  for j in np.arange(self.col)]
        self.k_space = np.zeros((self.row,self.col,3))
        self.fourier = np.zeros((self.row,self.col),dtype=complex)

        for i in range(self.row):
            for j in range(self.col):
                self.M[i][j] = np.ravel(
                    np.dot(self.Ry(m.radians(90)), self.rgbImage[i][j]))
                
        # print(self.M)

        self.phaseGradient()

    def phaseGradient(self):
        PixelSummation = [0, 0, 0]
        self.phaseValue = 360 / self.row
        for i1 in range (self.row):    #120
            self.rowangle = i1*360/self.row

            for j1 in range (self.col):
                PixelSummation = [0, 0, 0]
                self.colangle = j1*360/self.row
                for i in range (self.row):
                    for j in range (self.col):
                        self.M1[i][j] = np.ravel([np.abs(self.M[i][j][0])*m.cos(m.radians((i*self.rowangle)+(j*self.colangle))), 
                                                np.abs(self.M[i][j][0])*m.sin(m.radians((i*self.rowangle)+(j*self.colangle))), 0])
                        PixelSummation = [sum(cnt) for cnt in zip(PixelSummation, self.M1[i][j])]
                self.k_space[i1][j1] = PixelSummation
                self.fourier[i1][j1] = complex(PixelSummation[1],PixelSummation[0])
            self.generat_kspace()
                
        # print(self.fourier)

    def generat_phantom(self):
        self.phantom_flag = True
        self.figure_phantom.clear()
        result = np.fft.ifft2(self.fourier)
        # print(result)
        ax = self.figure_phantom.add_subplot(111)
        ax.imshow(np.abs(result), cmap='gray')
        self.canvas_phantom.draw()

    def generat_kspace(self):
        self.figure_kspace.clear()
        # self.ft_shifted = np.fft.fftshift(magnitude_spectrum)
        # self.ft_shifted2 = np.fft.fftshift(magnitude_spectrum)
        
        ax = self.figure_kspace.add_subplot(111)
        
        # for cnt in range(self.row-1,-1,-1):
        magnitude_spectrum = np.abs(self.fourier)

        # self.ft_shifted2 = np.fft.fft(magnitude_spectrum)
    # print(cnt)
        # magnitude_spectrum[:cnt] = 0
        ax.imshow(magnitude_spectrum, cmap='gray')
        self.canvas_kspace.draw()
        self.canvas_kspace.start_event_loop(0.0001)

        # self.ft_shifted2 = np.fft.fftshift(magnitude_spectrum)
        # ax.imshow(np.abs(self.ft_shifted2), cmap='gray')
        # self.canvas_kspace.draw()
    def resetSliders(self):
        # get the current value of the slider
        brughtness_slider_Value = self.ui.slider_brightness.value()
        contrast_slider_Value = self.ui.slider_contrast.value()
        # check if the current value is not equal to the initial value
        if (contrast_slider_Value & brughtness_slider_Value) != 100:
            # set the value of the slider to its initial value
            self.ui.label_15.setText("0")
            self.ui.label_9.setText("0")

            self.ui.slider_brightness.setValue(100)
            self.ui.slider_contrast.setValue(100)

        
    def adjust_contrast(self, value):
        try:
            if self.phantom_flag == False:
                self.figure_phantom.clear()
                self.canvas_phantom.draw()
            # self.clearFigures()
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
            if self.phantom_flag == False:
                self.figure_phantom.clear()
                self.canvas_phantom.draw()
            # self.clearFigures()

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

    def upload(self):
        self.figure_sequence.clear()
        self.data = json.load(open("data.json", "r"))
        axs = self.figure_sequence.subplots(4,sharex=True)
        self.figure_sequence.suptitle('Sequence')
        self.x_axis = np.linspace(-50, 50, 500)
        for dictionary in self.data:
            if dictionary['name'] == "RF_Pulse":
                self.RF_duration = np.linspace(dictionary["RF_Starting"],dictionary["RF_Ending"], 500)
                self.RF_amplitude = dictionary["RF_amplitude"] * \
                    abs(np.sinc(self.RF_duration))
                # print(self.RF_amplitude)
                axs[0].plot(self.x_axis, self.RF_amplitude, color='red')
                axs[0].set_ylabel('RF', fontsize=14,
                                  fontweight='bold', rotation=0, labelpad=20)
                axs[0].set_frame_on(False)
                axs[0].xaxis.set_visible(False)
                axs[0].axhline(y=0, color='black')
                axs[0].tick_params(axis='y', colors='red')

            if dictionary['name'] == "Gx_pulse":
                # self.Gx_duration = np.linspace(-50, 50, 500)
                self.Gx_amplitude = (np.where(self.x_axis >= dictionary["Gx_Starting"], dictionary["Gx_amplitude"], 0) 
                                    * np.where(self.x_axis < dictionary["Gx_Ending"], dictionary["Gx_amplitude"], np.where(self.x_axis == 1, 0.5, 0)))
                axs[1].plot(self.x_axis, self.Gx_amplitude, color='green')
                axs[1].set_ylabel('Gx', fontsize=14,
                                  fontweight='bold', rotation=0, labelpad=20)
                axs[1].set_frame_on(False)
                axs[1].xaxis.set_visible(False)
                axs[1].axhline(y=0, color='black')
                axs[1].tick_params(axis='y', colors='green')

            if dictionary['name'] == "Gy_Pulse":
                # print(len(self.RF_duration))
                # self.Gy_duration = np.linspace(-50, 50, 500)
                for i in range(0,5):
                    print(i)
                    self.Gy_amplitude = (np.where(self.x_axis >=  dictionary["Gy_Starting"], i, 0) 
                                        * np.where(self.x_axis < dictionary["Gy_Ending"], 1, 0))
                    axs[2].plot(self.x_axis, self.Gy_amplitude, color='blue')

                # axs[2].plot(self.x_axis, self.Gx_amplitude, color='blue')
                axs[2].set_ylabel('Gy', fontsize=14,
                                  fontweight='bold', rotation=0, labelpad=20)
                axs[2].set_frame_on(False)
                axs[2].xaxis.set_visible(False)
                axs[2].axhline(y=0, color='black')
                axs[2].tick_params(axis='y', colors='blue')

            if dictionary['name'] == "Readout":
                self.RO_duration = np.linspace(dictionary["Readout_Starting"], dictionary["Readout_Ending"], 500)
                self.RO_amplitude = (np.sinc(self.RO_duration))
                axs[3].plot(self.x_axis, self.RO_amplitude, color='brown')
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
            
            img_copy = self.image_orignal
            rgb_array = qimage2ndarray.rgb_view(self.image_orignal)

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