# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_classic.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 785)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("SheppLogan_Phantom256.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("\n"
"/*-----QWidget-----*/\n"
"QWidget\n"
"{\n"
"    background-color: #fff;\n"
"    color: red;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QLabel-----*/\n"
"QLabel\n"
"{\n"
"    background-color: transparent;\n"
"    color: #454544;\n"
"    font-weight: bold;\n"
"    font-size: 13px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QPushButton-----*/\n"
"QPushButton\n"
"{\n"
"    background-color: #5c55e9;\n"
"    color: #fff;\n"
"    font-size: 13px;\n"
"    font-weight: bold;\n"
"    border-top-right-radius: 15px;\n"
"    border-top-left-radius: 0px;\n"
"    border-bottom-right-radius: 0px;\n"
"    border-bottom-left-radius: 15px;\n"
"    padding: 10px;\n"
"\n"
"}\n"
"\n"
"\n"
"QPushButton::disabled\n"
"{\n"
"    background-color: #5c5c5c;\n"
"\n"
"}\n"
"\n"
"\n"
"QPushButton::hover\n"
"{\n"
"    background-color: #5564f2;\n"
"\n"
"}\n"
"\n"
"\n"
"QPushButton::pressed\n"
"{\n"
"    background-color: #3d4ef2;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QCheckBox-----*/\n"
"QCheckBox\n"
"{\n"
"    background-color: transparent;\n"
"    color: #5c55e9;\n"
"    font-size: 10px;\n"
"    font-weight: bold;\n"
"    border: none;\n"
"    border-radius: 5px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QCheckBox-----*/\n"
"QCheckBox::indicator\n"
"{\n"
"    background-color: #323232;\n"
"    border: 1px solid darkgray;\n"
"    width: 12px;\n"
"    height: 12px;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::indicator:checked\n"
"{\n"
"    image:url(\"./ressources/check.png\");\n"
"    background-color: #5c55e9;\n"
"    border: 1px solid #5c55e9;\n"
"\n"
"}\n"
"\n"
"\n"
"QCheckBox::indicator:unchecked:hover\n"
"{\n"
"    border: 1px solid #5c55e9;\n"
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
"\n"
"/*-----QLineEdit-----*/\n"
"QLineEdit\n"
"{\n"
"    background-color: #c2c7d5;\n"
"    color: #2a547f;\n"
"    border: none;\n"
"    padding: 5px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QListView-----*/\n"
"QListView\n"
"{\n"
"    background-color: #5c55e9;\n"
"    color: #fff;\n"
"    font-size: 14px;\n"
"    font-weight: bold;\n"
"    show-decoration-selected: 0;\n"
"    border-radius: 4px;\n"
"    padding-left: -15px;\n"
"    padding-right: -15px;\n"
"    padding-top: 5px;\n"
"\n"
"} \n"
"\n"
"\n"
"QListView:disabled \n"
"{\n"
"    background-color: #5c5c5c;\n"
"\n"
"}\n"
"\n"
"\n"
"QListView::item\n"
"{\n"
"    background-color: #454e5e;\n"
"    border: none;\n"
"    padding: 10px;\n"
"    border-radius: 0px;\n"
"    padding-left : 10px;\n"
"    height: 32px;\n"
"\n"
"}\n"
"\n"
"\n"
"QListView::item:selected\n"
"{\n"
"    color: #000;\n"
"    background-color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QListView::item:!selected\n"
"{\n"
"    color:white;\n"
"    background-color: transparent;\n"
"    border: none;\n"
"    padding-left : 10px;\n"
"\n"
"}\n"
"\n"
"\n"
"QListView::item:!selected:hover\n"
"{\n"
"    color: #fff;\n"
"    background-color: #5564f2;\n"
"    border: none;\n"
"    padding-left : 10px;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QTreeView-----*/\n"
"QTreeView \n"
"{\n"
"    background-color: #fff;\n"
"    show-decoration-selected: 0;\n"
"    color: #454544;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView:disabled\n"
"{\n"
"    background-color: #242526;\n"
"    show-decoration-selected: 0;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::item \n"
"{\n"
"    border-top-color: transparent;\n"
"    border-bottom-color: transparent;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::item:hover \n"
"{\n"
"    background-color: #bcbdbb;\n"
"    color: #000;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::item:selected \n"
"{\n"
"    background-color: #5c55e9;\n"
"    color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::item:selected:active\n"
"{\n"
"    background-color: #5c55e9;\n"
"    color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::item:selected:disabled\n"
"{\n"
"    background-color: #525251;\n"
"    color: #656565;\n"
"\n"
"}\n"
"\n"
"\n"
"QTreeView::branch:has-children:!has-siblings:closed,\n"
"QTreeView::branch:closed:has-children:has-siblings \n"
"{\n"
"    image: url(://tree-closed.png);\n"
"\n"
"}\n"
"\n"
"QTreeView::branch:open:has-children:!has-siblings,\n"
"QTreeView::branch:open:has-children:has-siblings  \n"
"{\n"
"    image: url(://tree-open.png);\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QTableView & QTableWidget-----*/\n"
"QTableView\n"
"{\n"
"    background-color: #fff;\n"
"    border: 1px solid gray;\n"
"    color: #454544;\n"
"    gridline-color: gray;\n"
"    outline : 0;\n"
"\n"
"}\n"
"\n"
"\n"
"QTableView::disabled\n"
"{\n"
"    background-color: #242526;\n"
"    border: 1px solid #32414B;\n"
"    color: #656565;\n"
"    gridline-color: #656565;\n"
"    outline : 0;\n"
"\n"
"}\n"
"\n"
"\n"
"QTableView::item:hover \n"
"{\n"
"    background-color: #bcbdbb;\n"
"    color: #000;\n"
"\n"
"}\n"
"\n"
"\n"
"QTableView::item:selected \n"
"{\n"
"    background-color: #5c55e9;\n"
"    color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QTableView::item:selected:disabled\n"
"{\n"
"    background-color: #1a1b1c;\n"
"    border: 2px solid #525251;\n"
"    color: #656565;\n"
"\n"
"}\n"
"\n"
"\n"
"QTableCornerButton::section\n"
"{\n"
"    background-color: #ced5e3;\n"
"    border: none;\n"
"    color: #fff;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section\n"
"{\n"
"    color: #2a547f;\n"
"    border: 0px;\n"
"    background-color: #ced5e3;\n"
"    padding: 5px;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section:disabled\n"
"{\n"
"    background-color: #525251;\n"
"    color: #656565;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section:checked\n"
"{\n"
"    color: #fff;\n"
"    background-color: #5c55e9;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section:checked:disabled\n"
"{\n"
"    color: #656565;\n"
"    background-color: #525251;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section::vertical::first,\n"
"QHeaderView::section::vertical::only-one\n"
"{\n"
"    border-top: 1px solid #353635;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section::vertical\n"
"{\n"
"    border-top: 1px solid #353635;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section::horizontal::first,\n"
"QHeaderView::section::horizontal::only-one\n"
"{\n"
"    border-left: 1px solid #353635;\n"
"\n"
"}\n"
"\n"
"\n"
"QHeaderView::section::horizontal\n"
"{\n"
"    border-left: 1px solid #353635;\n"
"\n"
"}\n"
"\n"
"\n"
"/*-----QScrollBar-----*/\n"
"QScrollBar:horizontal \n"
"{\n"
"    background-color: transparent;\n"
"    height: 8px;\n"
"    margin: 0px;\n"
"    padding: 0px;\n"
"\n"
"}\n"
"\n"
"\n"
"QScrollBar::handle:horizontal \n"
"{\n"
"    border: none;\n"
"    min-width: 100px;\n"
"    background-color: #7e92b7;\n"
"\n"
"}\n"
"\n"
"\n"
"QScrollBar::add-line:horizontal, \n"
"QScrollBar::sub-line:horizontal,\n"
"QScrollBar::add-page:horizontal, \n"
"QScrollBar::sub-page:horizontal \n"
"{\n"
"    width: 0px;\n"
"    background-color: #d8dce6;\n"
"\n"
"}\n"
"\n"
"\n"
"QScrollBar:vertical \n"
"{\n"
"    background-color: transparent;\n"
"    width: 8px;\n"
"    margin: 0;\n"
"\n"
"}\n"
"\n"
"\n"
"QScrollBar::handle:vertical \n"
"{\n"
"    border: none;\n"
"    min-height: 100px;\n"
"    background-color: #7e92b7;\n"
"\n"
"}\n"
"\n"
"\n"
"QScrollBar::add-line:vertical, \n"
"QScrollBar::sub-line:vertical,\n"
"QScrollBar::add-page:vertical, \n"
"QScrollBar::sub-page:vertical \n"
"{\n"
"    height: 0px;\n"
"    background-color: #d8dce6;\n"
"\n"
"}\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setSpacing(5)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.verticalLayout_Phantom = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_Phantom.setObjectName("verticalLayout_Phantom")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.pushButton_Browse = QtWidgets.QPushButton(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Browse.sizePolicy().hasHeightForWidth())
        self.pushButton_Browse.setSizePolicy(sizePolicy)
        self.pushButton_Browse.setObjectName("pushButton_Browse")
        self.horizontalLayout_9.addWidget(self.pushButton_Browse)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_7 = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.comboBox_size = QtWidgets.QComboBox(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_size.sizePolicy().hasHeightForWidth())
        self.comboBox_size.setSizePolicy(sizePolicy)
        self.comboBox_size.setObjectName("comboBox_size")
        self.horizontalLayout.addWidget(self.comboBox_size)
        self.horizontalLayout_7.addLayout(self.horizontalLayout)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout_5.addWidget(self.label)
        self.comboBox_property = QtWidgets.QComboBox(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_property.sizePolicy().hasHeightForWidth())
        self.comboBox_property.setSizePolicy(sizePolicy)
        self.comboBox_property.setObjectName("comboBox_property")
        self.horizontalLayout_5.addWidget(self.comboBox_property)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_7)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_14 = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_3.addWidget(self.label_14)
        self.slider_brightness = QtWidgets.QSlider(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
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
        self.horizontalLayout_3.addWidget(self.slider_brightness)
        self.label_15 = QtWidgets.QLabel(self.widget)
        self.label_15.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_3.addWidget(self.label_15)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_8 = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_2.addWidget(self.label_8)
        self.slider_contrast = QtWidgets.QSlider(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
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
        self.horizontalLayout_2.addWidget(self.slider_contrast)
        self.label_9 = QtWidgets.QLabel(self.widget)
        self.label_9.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_2.addWidget(self.label_9)
        self.verticalLayout_4.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_8.addLayout(self.verticalLayout_4)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_8)
        self.verticalLayout_Phantom.addLayout(self.horizontalLayout_9)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_view_img = QtWidgets.QLabel(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_view_img.sizePolicy().hasHeightForWidth())
        self.label_view_img.setSizePolicy(sizePolicy)
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
        self.verticalLayout_5.addWidget(self.label_view_img)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.lineEdit_t1 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_t1.setEnabled(False)
        self.lineEdit_t1.setObjectName("lineEdit_t1")
        self.verticalLayout_3.addWidget(self.lineEdit_t1)
        self.horizontalLayout_6.addLayout(self.verticalLayout_3)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.lineEdit_t2 = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_t2.setEnabled(False)
        self.lineEdit_t2.setObjectName("lineEdit_t2")
        self.verticalLayout_2.addWidget(self.lineEdit_t2)
        self.horizontalLayout_6.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.lineEdit_sd = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_sd.setEnabled(False)
        self.lineEdit_sd.setObjectName("lineEdit_sd")
        self.verticalLayout.addWidget(self.lineEdit_sd)
        self.horizontalLayout_6.addLayout(self.verticalLayout)
        self.verticalLayout_5.addLayout(self.horizontalLayout_6)
        self.verticalLayout_Phantom.addLayout(self.verticalLayout_5)
        self.horizontalLayout_10.addWidget(self.widget)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_10.addWidget(self.line)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setObjectName("widget1")
        self.verticalLayout_Kspace = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_Kspace.setObjectName("verticalLayout_Kspace")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_5 = QtWidgets.QLabel(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("background-color: qlineargradient(spread:pad, x1:1, y1:0.493909, x2:0, y2:0.511045, stop:0.0547264 rgba(211, 138, 44, 0), stop:0.462687 rgba(92, 85, 233, 255), stop:0.950249 rgba(211, 138, 44, 0));\n"
"\n"
"color: #fff;")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_7.addWidget(self.label_5)
        self.horizontalLayout_Kspace = QtWidgets.QHBoxLayout()
        self.horizontalLayout_Kspace.setObjectName("horizontalLayout_Kspace")
        self.verticalLayout_7.addLayout(self.horizontalLayout_Kspace)
        self.label_6 = QtWidgets.QLabel(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("background-color: qlineargradient(spread:pad, x1:1, y1:0.493909, x2:0, y2:0.511045, stop:0.0547264 rgba(211, 138, 44, 0), stop:0.462687 rgba(92, 85, 233, 255), stop:0.950249 rgba(211, 138, 44, 0));\n"
"\n"
"color: #fff;")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_7.addWidget(self.label_6)
        self.horizontalLayout_phantom = QtWidgets.QHBoxLayout()
        self.horizontalLayout_phantom.setObjectName("horizontalLayout_phantom")
        self.verticalLayout_7.addLayout(self.horizontalLayout_phantom)
        self.verticalLayout_Kspace.addLayout(self.verticalLayout_7)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pushButton_Run = QtWidgets.QPushButton(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_Run.setFont(font)
        self.pushButton_Run.setObjectName("pushButton_Run")
        self.horizontalLayout_4.addWidget(self.pushButton_Run)
        self.pushButton_Run_Generat_phantom = QtWidgets.QPushButton(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_Run_Generat_phantom.setFont(font)
        self.pushButton_Run_Generat_phantom.setObjectName("pushButton_Run_Generat_phantom")
        self.horizontalLayout_4.addWidget(self.pushButton_Run_Generat_phantom)
        self.verticalLayout_Kspace.addLayout(self.horizontalLayout_4)
        self.verticalLayout_Kspace.setStretch(0, 1)
        self.horizontalLayout_10.addWidget(self.widget1)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout_10.addWidget(self.line_2)
        self.widget2 = QtWidgets.QWidget(self.centralwidget)
        self.widget2.setObjectName("widget2")
        self.verticalLayout_Sequence = QtWidgets.QVBoxLayout(self.widget2)
        self.verticalLayout_Sequence.setObjectName("verticalLayout_Sequence")
        self.label_11 = QtWidgets.QLabel(self.widget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.label_11.setFont(font)
        self.label_11.setStyleSheet("background-color: qlineargradient(spread:pad, x1:1, y1:0.493909, x2:0, y2:0.511045, stop:0.0547264 rgba(211, 138, 44, 0), stop:0.462687 rgba(92, 85, 233, 255), stop:0.950249 rgba(211, 138, 44, 0));\n"
"\n"
"color: #fff;")
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_Sequence.addWidget(self.label_11)
        self.horizontalLayout_Sequence = QtWidgets.QHBoxLayout()
        self.horizontalLayout_Sequence.setObjectName("horizontalLayout_Sequence")
        self.verticalLayout_Sequence.addLayout(self.horizontalLayout_Sequence)
        self.pushButton_Plot = QtWidgets.QPushButton(self.widget2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_Plot.sizePolicy().hasHeightForWidth())
        self.pushButton_Plot.setSizePolicy(sizePolicy)
        self.pushButton_Plot.setObjectName("pushButton_Plot")
        self.verticalLayout_Sequence.addWidget(self.pushButton_Plot)
        self.horizontalLayout_10.addWidget(self.widget2)
        self.horizontalLayout_10.setStretch(2, 1)
        self.horizontalLayout_10.setStretch(4, 1)
        self.gridLayout.addLayout(self.horizontalLayout_10, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.slider_brightness.sliderMoved['int'].connect(self.label_15.setNum)
        self.slider_contrast.sliderMoved['int'].connect(self.label_9.setNum)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Reconstruction"))
        self.pushButton_Browse.setText(_translate("MainWindow", "Browse"))
        self.label_7.setText(_translate("MainWindow", "size"))
        self.label.setText(_translate("MainWindow", "property"))
        self.label_14.setText(_translate("MainWindow", "Brightness"))
        self.label_15.setText(_translate("MainWindow", "0"))
        self.label_8.setText(_translate("MainWindow", "Contrast"))
        self.label_9.setText(_translate("MainWindow", "0"))
        self.label_2.setText(_translate("MainWindow", "T1"))
        self.label_3.setText(_translate("MainWindow", "T2"))
        self.label_4.setText(_translate("MainWindow", "SD"))
        self.label_5.setText(_translate("MainWindow", "K-Space"))
        self.label_6.setText(_translate("MainWindow", "Phantom from K-space"))
        self.pushButton_Run.setText(_translate("MainWindow", "Run K-space"))
        self.pushButton_Run_Generat_phantom.setText(_translate("MainWindow", "Generate Phantom"))
        self.label_11.setText(_translate("MainWindow", "Sequence"))
        self.pushButton_Plot.setText(_translate("MainWindow", "Plot"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
