# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.8
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(814, 859)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\SheppLogan_Phantom256.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("\n"
"\n"
"/*-----QWidget-----*/\n"
"QWidget\n"
"{\n"
"    color: #000000;\n"
"    border-color: #000000;\n"
"    background-color: rgb(16, 10, 3);\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QLabel-----*/\n"
"QLabel\n"
"{\n"
"\n"
"font-family: \'Josefin Sans\', sans-serif;\n"
"font-size: 9pt;\n"
"font-weight: 500;\n"
"letter-spacing: 0.22rem;\n"
"font-style: oblique;\n"
"text-transform: capitalize;\n"
"color: #FFFFFF;\n"
"border-radius: 1 px;\n"
"border-style: solid;\n"
"border-color: #000000;\n"
"    background-color: transparent;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QPushButton-----*/\n"
"QPushButton\n"
"{\n"
"    color: #fff;\n"
"    background-color: rgba(80, 255, 130, 200);\n"
"    font-weight: bold;\n"
"    border-style: solid;\n"
"    border-width: 1px;\n"
"    border-radius: 17px;\n"
"    border-color: #000;\n"
"    padding: 10px;\n"
"\n"
"}\n"
"\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color: lightgrey;\n"
"    color: #000;\n"
"\n"
"}\n"
"\n"
"\n"
"QPushButton::pressed\n"
"{\n"
"    background-color: lightgreen;\n"
"    color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QToolButton-----*/\n"
"QToolButton\n"
"{\n"
"    background-color: #292f45;\n"
"    color: #000000;\n"
"    border-style: solid;\n"
"    border-color: #000000;\n"
"\n"
"}\n"
"\n"
"\n"
"QToolButton::hover\n"
"{\n"
"    background-color: #fc7c11;\n"
"    color: #000000;\n"
"    padding: 2px;\n"
"    border-radius: 15px;\n"
"    border-color: #fc7c11;\n"
"\n"
"}\n"
"\n"
"\n"
"QToolButton::pressed\n"
"{\n"
"    background-color: #fc7c11;\n"
"    color: #000000;\n"
"    border-style: solid;\n"
"    border-width: 2px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QLineEdit-----*/\n"
"QLineEdit{\n"
"    background-color: #292f45;\n"
"    color: #b9b9bb;\n"
"    font-weight: bold;\n"
"    border-style: solid;\n"
"    border-width: 2px;\n"
"    border-top: 0px;\n"
"    border-left: 0px;\n"
"    border-right: 0px;\n"
"    border-color: #b9b9bb;\n"
"    padding: 10px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QCheckBox-----*/\n"
"QCheckBox\n"
"{\n"
"    background-color: transparent;\n"
"    color: #b9b9bb;\n"
"    font-weight: bold;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::indicator\n"
"{\n"
"    color: #b1b1b1;\n"
"    background-color: #00111d;\n"
"    border: 1px solid #f0742f;\n"
"    width: 12px;\n"
"    height: 12px;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::indicator:checked\n"
"{\n"
"    image:url(\"./ressources/check.png\"); /*To replace*/\n"
"    background-color: #1f2b2b;\n"
"    border: 1px solid #f0742f;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::indicator:unchecked:hover\n"
"{\n"
"    border: 1px solid #f0742f;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::disabled\n"
"{\n"
"    color: #656565;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::indicator:disabled\n"
"{\n"
"    background-color: #656565;\n"
"    color: #656565;\n"
"    border: 1px solid #656565;\n"
"\n"
"}\n"
"\n"
"QRadioButton\n"
"{\n"
"    \n"
"font-family: \'Josefin Sans\', sans-serif;\n"
"font-size: 8pt;\n"
"font-weight: 500;\n"
"    spacing: 5px;\n"
"    outline: none;\n"
"    color: #bbb;\n"
"    margin-bottom: 2px;\n"
"}\n"
"\n"
"QRadioButton:disabled\n"
"{\n"
"    color: #777777;\n"
"}\n"
"\n"
"QTabWidget{\n"
"font: 72pt \"MS Shell Dlg 2\";\n"
"}\n"
"QTabBar {\n"
"\n"
"  background-color: transparent;\n"
"  height: 30px;\n"
"    font-weight: 500;\n"
"    font-size: 10pt;\n"
"    \n"
"    font-family: \"Segoe Print\";\n"
"}\n"
"\n"
"QTabBar::tab {\n"
"border: 1px solid grey;\n"
"border-radius: 12px ;\n"
"background-color: rgba(236, 236, 236, 200);\n"
"\n"
"  border: 1px solid #CCCCCC;\n"
"  color: #333333;\n"
"  padding: 5px 10px;\n"
"  margin: 3px;\n"
" \n"
"}\n"
"\n"
"QTabWidget::tab-bar{alignment:center;}\n"
"\n"
"QTabBar::tab:!selected {\n"
"    \n"
"  color: #FFFFFF;\n"
"    background-color: qlineargradient(spread:repeat, x1:1, y1:0, x2:1, y2:1, stop:0 rgba(234, 208, 174, 255), stop:0.472637 rgba(152, 97, 0, 255));\n"
"}\n"
"\n"
"QTabBar::tab:hover{\n"
"background-color: rgba(245, 245, 245, 120);\n"
"\n"
"}\n"
"\n"
"/*-----QComboBox-----*/\n"
"QComboBox\n"
"{\n"
"    padding-left: 6px;\n"
"    border: 1px solid #1d1d1d;\n"
"    background-color: rgb(158, 101, 1);\n"
"    color: #fff;\n"
"    height: 20px;\n"
"\n"
"}\n"
"\n"
"\n"
"QComboBox:on\n"
"{\n"
"    background-color: transparent;\n"
"    color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QComboBox QAbstractItemView\n"
"{\n"
"    background-color: transparent;\n"
"    color: #fff;\n"
"    selection-background-color: #fea732;\n"
"    selection-color: #000;\n"
"    outline: 0;\n"
"\n"
"}\n"
"\n"
"QSlider::groove:horizontal {\n"
"    background-color: #dddddd;\n"
"    border: 1px solid #aaaaaa;\n"
"    height: 8px;\n"
"    border-radius: 4px;\n"
"}\n"
"\n"
"QSlider::handle:horizontal {\n"
"    background-color: #fea735;\n"
"    border: 1px solid #aaaaaa;\n"
"    width: 16px;\n"
"    height: 16px;\n"
"    margin: -4px 0;\n"
"    border-radius: 8px;\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(72)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        font.setKerning(True)
        self.tabWidget.setFont(font)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setMovable(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_phantom = QtWidgets.QWidget()
        self.tab_phantom.setObjectName("tab_phantom")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_phantom)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setContentsMargins(-1, 0, -1, -1)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.line = QtWidgets.QFrame(self.tab_phantom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line.sizePolicy().hasHeightForWidth())
        self.line.setSizePolicy(sizePolicy)
        self.line.setStyleSheet("Line{\n"
"background-color: qlineargradient(spread:pad, x1:1, y1:0.493909, x2:0, y2:0.511045, stop:0.0547264 rgba(211, 138, 44, 60), stop:0.462687 rgba(238, 152, 0, 255), stop:0.950249 rgba(211, 138, 44, 60));\n"
"\n"
"}")
        self.line.setLineWidth(0)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_5.addWidget(self.line)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_Browse = QtWidgets.QPushButton(self.tab_phantom)
        self.pushButton_Browse.setObjectName("pushButton_Browse")
        self.horizontalLayout_3.addWidget(self.pushButton_Browse)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_7 = QtWidgets.QLabel(self.tab_phantom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Josefin Sans,sans-serif")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(62)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_5.addWidget(self.label_7)
        self.comboBox_size = QtWidgets.QComboBox(self.tab_phantom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_size.sizePolicy().hasHeightForWidth())
        self.comboBox_size.setSizePolicy(sizePolicy)
        self.comboBox_size.setObjectName("comboBox_size")
        self.horizontalLayout_5.addWidget(self.comboBox_size)
        self.horizontalLayout_5.setStretch(1, 1)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.tab_phantom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Josefin Sans,sans-serif")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(62)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.comboBox_property = QtWidgets.QComboBox(self.tab_phantom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_property.sizePolicy().hasHeightForWidth())
        self.comboBox_property.setSizePolicy(sizePolicy)
        self.comboBox_property.setObjectName("comboBox_property")
        self.horizontalLayout.addWidget(self.comboBox_property)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(0, 0, -1, -1)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setContentsMargins(20, 0, -1, -1)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setContentsMargins(0, -1, -1, -1)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_8 = QtWidgets.QLabel(self.tab_phantom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Josefin Sans,sans-serif")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(62)
        self.label_8.setFont(font)
        self.label_8.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_7.addWidget(self.label_8)
        self.slider_contrast = QtWidgets.QSlider(self.tab_phantom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slider_contrast.sizePolicy().hasHeightForWidth())
        self.slider_contrast.setSizePolicy(sizePolicy)
        self.slider_contrast.setMinimum(50)
        self.slider_contrast.setMaximum(150)
        self.slider_contrast.setSingleStep(10)
        self.slider_contrast.setProperty("value", 100)
        self.slider_contrast.setSliderPosition(100)
        self.slider_contrast.setOrientation(QtCore.Qt.Horizontal)
        self.slider_contrast.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.slider_contrast.setTickInterval(0)
        self.slider_contrast.setObjectName("slider_contrast")
        self.horizontalLayout_7.addWidget(self.slider_contrast)
        self.label_9 = QtWidgets.QLabel(self.tab_phantom)
        self.label_9.setEnabled(False)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_7.addWidget(self.label_9)
        self.horizontalLayout_7.setStretch(1, 1)
        self.verticalLayout_9.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setContentsMargins(0, -1, -1, -1)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_14 = QtWidgets.QLabel(self.tab_phantom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Josefin Sans,sans-serif")
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(62)
        self.label_14.setFont(font)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_10.addWidget(self.label_14)
        self.slider_brightness = QtWidgets.QSlider(self.tab_phantom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slider_brightness.sizePolicy().hasHeightForWidth())
        self.slider_brightness.setSizePolicy(sizePolicy)
        self.slider_brightness.setMinimum(0)
        self.slider_brightness.setMaximum(200)
        self.slider_brightness.setSingleStep(10)
        self.slider_brightness.setProperty("value", 100)
        self.slider_brightness.setSliderPosition(100)
        self.slider_brightness.setOrientation(QtCore.Qt.Horizontal)
        self.slider_brightness.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.slider_brightness.setTickInterval(0)
        self.slider_brightness.setObjectName("slider_brightness")
        self.horizontalLayout_10.addWidget(self.slider_brightness)
        self.label_15 = QtWidgets.QLabel(self.tab_phantom)
        self.label_15.setEnabled(False)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_10.addWidget(self.label_15)
        self.horizontalLayout_10.setStretch(1, 1)
        self.verticalLayout_9.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_6.addLayout(self.verticalLayout_9)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_3.setStretch(1, 1)
        self.horizontalLayout_3.setStretch(2, 1)
        self.horizontalLayout_3.setStretch(3, 2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_view_img = QtWidgets.QLabel(self.tab_phantom)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_view_img.sizePolicy().hasHeightForWidth())
        self.label_view_img.setSizePolicy(sizePolicy)
        self.label_view_img.setMaximumSize(QtCore.QSize(1920, 720))
        self.label_view_img.setMouseTracking(True)
        self.label_view_img.setStyleSheet("QLabel{\n"
"border: 1px solid lightgrey;\n"
"\n"
"}")
        self.label_view_img.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_view_img.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_view_img.setText("")
        self.label_view_img.setScaledContents(True)
        self.label_view_img.setObjectName("label_view_img")
        self.verticalLayout_4.addWidget(self.label_view_img)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.tab_phantom)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.lineEdit_t1 = QtWidgets.QLineEdit(self.tab_phantom)
        self.lineEdit_t1.setEnabled(False)
        self.lineEdit_t1.setObjectName("lineEdit_t1")
        self.verticalLayout.addWidget(self.lineEdit_t1)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.tab_phantom)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.lineEdit_t2 = QtWidgets.QLineEdit(self.tab_phantom)
        self.lineEdit_t2.setEnabled(False)
        self.lineEdit_t2.setObjectName("lineEdit_t2")
        self.verticalLayout_2.addWidget(self.lineEdit_t2)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.tab_phantom)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_3.addWidget(self.label_4)
        self.lineEdit_sd = QtWidgets.QLineEdit(self.tab_phantom)
        self.lineEdit_sd.setEnabled(False)
        self.lineEdit_sd.setObjectName("lineEdit_sd")
        self.verticalLayout_3.addWidget(self.lineEdit_sd)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)
        self.gridLayout_4.addLayout(self.verticalLayout_5, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_phantom, "")
        self.tab_K = QtWidgets.QWidget()
        self.tab_K.setObjectName("tab_K")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_K)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pushButton_Run = QtWidgets.QPushButton(self.tab_K)
        self.pushButton_Run.setObjectName("pushButton_Run")
        self.horizontalLayout_4.addWidget(self.pushButton_Run)
        self.pushButton_Run_Generat_phantom = QtWidgets.QPushButton(self.tab_K)
        self.pushButton_Run_Generat_phantom.setObjectName("pushButton_Run_Generat_phantom")
        self.horizontalLayout_4.addWidget(self.pushButton_Run_Generat_phantom)
        self.gridLayout_3.addLayout(self.horizontalLayout_4, 1, 0, 1, 1)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_5 = QtWidgets.QLabel(self.tab_K)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setStyleSheet("background-color: qlineargradient(spread:pad, x1:1, y1:0.493909, x2:0, y2:0.511045, stop:0.0547264 rgba(211, 138, 44, 0), stop:0.462687 rgba(238, 152, 0, 255), stop:0.950249 rgba(211, 138, 44, 0));")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_7.addWidget(self.label_5)
        self.horizontalLayout_Kspace = QtWidgets.QHBoxLayout()
        self.horizontalLayout_Kspace.setObjectName("horizontalLayout_Kspace")
        self.verticalLayout_7.addLayout(self.horizontalLayout_Kspace)
        self.label_6 = QtWidgets.QLabel(self.tab_K)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setStyleSheet("background-color: qlineargradient(spread:pad, x1:1, y1:0.493909, x2:0, y2:0.511045, stop:0.0547264 rgba(211, 138, 44, 0), stop:0.462687 rgba(238, 152, 0, 255), stop:0.950249 rgba(211, 138, 44, 0));")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_7.addWidget(self.label_6)
        self.horizontalLayout_phantom = QtWidgets.QHBoxLayout()
        self.horizontalLayout_phantom.setObjectName("horizontalLayout_phantom")
        self.verticalLayout_7.addLayout(self.horizontalLayout_phantom)
        self.gridLayout_3.addLayout(self.verticalLayout_7, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_K, "")
        self.tab_Seq = QtWidgets.QWidget()
        self.tab_Seq.setObjectName("tab_Seq")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_Seq)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.pushButton_Plot = QtWidgets.QPushButton(self.tab_Seq)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Plot.sizePolicy().hasHeightForWidth())
        self.pushButton_Plot.setSizePolicy(sizePolicy)
        self.pushButton_Plot.setObjectName("pushButton_Plot")
        self.verticalLayout_6.addWidget(self.pushButton_Plot)
        self.horizontalLayout_Sequence = QtWidgets.QHBoxLayout()
        self.horizontalLayout_Sequence.setObjectName("horizontalLayout_Sequence")
        self.verticalLayout_6.addLayout(self.horizontalLayout_Sequence)
        self.gridLayout_2.addLayout(self.verticalLayout_6, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_Seq, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.slider_contrast.sliderMoved['int'].connect(self.label_9.setNum) # type: ignore
        self.slider_brightness.sliderMoved['int'].connect(self.label_15.setNum) # type: ignore
        self.slider_contrast.valueChanged['int'].connect(self.label_9.setNum) # type: ignore
        self.slider_brightness.valueChanged['int'].connect(self.label_15.setNum) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Reconstruction"))
        self.pushButton_Browse.setText(_translate("MainWindow", "Browse"))
        self.label_7.setText(_translate("MainWindow", "size"))
        self.label.setText(_translate("MainWindow", "property"))
        self.label_8.setText(_translate("MainWindow", "Contrast"))
        self.label_9.setText(_translate("MainWindow", "0"))
        self.label_14.setText(_translate("MainWindow", "Brightness"))
        self.label_15.setText(_translate("MainWindow", "0"))
        self.label_2.setText(_translate("MainWindow", "T1"))
        self.label_3.setText(_translate("MainWindow", "T2"))
        self.label_4.setText(_translate("MainWindow", "SD"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_phantom), _translate("MainWindow", "Phantom"))
        self.pushButton_Run.setText(_translate("MainWindow", "Run K-space"))
        self.pushButton_Run_Generat_phantom.setText(_translate("MainWindow", "Generate Phantom"))
        self.label_5.setText(_translate("MainWindow", "K-Space"))
        self.label_6.setText(_translate("MainWindow", "Phantom from K-space"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_K), _translate("MainWindow", "K-Space"))
        self.pushButton_Plot.setText(_translate("MainWindow", "Plot"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_Seq), _translate("MainWindow", "Sequence"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
