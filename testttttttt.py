import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage , QColor, qRgba
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QSlider, QVBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create the main widget and set its layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Create the image label and add it to the layout
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        
        # Create the contrast slider and add it to the layout
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(0)
        self.contrast_slider.setMaximum(200)
        self.contrast_slider.setValue(100)
        self.layout.addWidget(self.contrast_slider)
        
        # Connect the slider to the contrast adjustment function
        self.contrast_slider.valueChanged.connect(self.adjust_contrast)
        
        # Load the image and display it
        self.image = QImage("normal.jpeg")
        print(type(self.image))
        self.image_label.setPixmap(QPixmap.fromImage(self.image))
        
    def adjust_contrast(self, value):
        # Calculate the contrast factor based on the slider value
        contrast_factor = (value + 50) / 100.0
        
        # Create a copy of the original image and apply the contrast adjustment
        adjusted_image = self.image.copy()
        for i in range(adjusted_image.width()):
            for j in range(adjusted_image.height()):
                pixel = adjusted_image.pixel(i, j)
                r, g, b, a = QColor(pixel).getRgb()
                r = int((r - 128) * contrast_factor + 128)
                g = int((g - 128) * contrast_factor + 128)
                b = int((b - 128) * contrast_factor + 128)
                adjusted_image.setPixel(i, j, qRgba(r, g, b, a))
        
        # Display the adjusted image
        self.image_label.setPixmap(QPixmap.fromImage(adjusted_image))
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
