import sys
from PyQt5.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsRectItem
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt

class MainView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.draw_lines()

    def draw_lines(self):
        pen = QPen(Qt.black)
        y = 50
        for i in range(5):
            line = self.scene.addLine(0, y, 200, y, pen)
            rect = QGraphicsRectItem(20, y-10, 40, 20)
            self.scene.addItem(rect)
            rect.setParentItem(line)
            y += 50

if __name__ == '__main__':
    app = QApplication(sys.argv)
    view = MainView()
    view.show()
    sys.exit(app.exec_())



'''import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage , QColor
from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit, QVBoxLayout, QWidget


class Example(QWidget):
    def __init__(self):
        super().__init__()

        # Create QLabel and QLineEdit widgets
        self.label = QLabel()
        self.line_edit = QLineEdit()

        # Create a QVBoxLayout and add the widgets to it
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        self.setLayout(layout)

        # Load an image into the label
        pixmap = QPixmap("SheppLogan_Phantom.png")
        self.label.setPixmap(pixmap)

        # Set the label to be clickable
        self.label.setMouseTracking(True)
        self.label.mousePressEvent = self.get_pixel_value

    def get_pixel_value(self, event):
        # Get the local position of the mouse click
        point = event.pos()

        # Get the pixel value of the clicked point
        image = self.label.pixmap().toImage()
        color = QColor(image.pixel(point))

        # Calculate the gray pixel density value
        gray_value = int(0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue())

        # Set the gray pixel density value to the line edit
        self.line_edit.setText(str(gray_value))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())
'''
'''import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage   , QColor
from PyQt5.QtWidgets import QApplication, QLabel, QLineEdit, QVBoxLayout, QWidget


class Example(QWidget):
    def __init__(self):
        super().__init__()

        # Create QLabel and QLineEdit widgets
        self.label = QLabel()
        self.line_edit = QLineEdit()

        # Create a QVBoxLayout and add the widgets to it
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        self.setLayout(layout)

        # Load an image into the label
        pixmap = QPixmap("SheppLogan_Phantom.png")
        self.label.setPixmap(pixmap)

        # Set the label to be clickable
        self.label.setMouseTracking(True)
        self.label.mousePressEvent = self.get_pixel_value

    def get_pixel_value(self, event):
        # Get the local position of the mouse click
        point = event.pos()

        # Get the pixel value of the clicked point
        image = self.label.pixmap().toImage()
        color = QColor(image.pixel(point))

        # Set the pixel value to the line edit
        self.line_edit.setText(f"R: {color.red()}, G: {color.green()}, B: {color.blue()}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())'''

'''      
    def t1(self, in_image):
        shepp_t1 = np.zeros((in_image.shape[0], in_image.shape[1]), dtype=np.uint8)
        for i in range(in_image.shape[0]):
            for j in range(in_image.shape[1]):
                if in_image[i][j] == 255:  # white matter
                    shepp_t1[i][j] = 10
                elif in_image[i][j] == 150:  # gray matter
                    shepp_t1[i][j] = 70
                elif in_image[i][j] == 90:  # fat
                    shepp_t1[i][j] = 150
                elif in_image[i][j] == 0:  # water
                    shepp_t1[i][j] = 255
                else:
                    shepp_t1[i][j] = 255
        # convert image to qimage
        shepp_t1 = qimage2ndarray.array2qimage(shepp_t1)
        return shepp_t1

    def t2(self, in_image):
        shepp_t2 = np.zeros(
            (in_image.shape[0], in_image.shape[1]), dtype=np.uint8)
        for i in range(in_image.shape[0]):
            for j in range(in_image.shape[1]):
                if in_image[i][j] == 255:  # white matter
                    shepp_t2[i][j] = 50
                elif in_image[i][j] == 150:  # gray matter
                    shepp_t2[i][j] = 255
                elif in_image[i][j] == 90:  # fat
                    shepp_t2[i][j] = 10
                elif in_image[i][j] == 0:  # water
                    shepp_t2[i][j] = 200
                else:
                    shepp_t2[i][j] = 200
        # convert image to qimage
        shepp_t2 = qimage2ndarray.array2qimage(shepp_t2)
        return shepp_t2

    def SD(self, in_image):
        shepp_SD = np.zeros(
            (in_image.shape[0], in_image.shape[1]), dtype=np.uint8)
        for i in range(in_image.shape[0]):
            for j in range(in_image.shape[1]):
                if in_image[i][j] == 255:  # white matter
                    shepp_SD[i][j] = 50
                elif in_image[i][j] == 150:  # gray matter
                    shepp_SD[i][j] = 140
                elif in_image[i][j] == 90:  # fat
                    shepp_SD[i][j] = 120
                elif in_image[i][j] == 0:  # water
                    shepp_SD[i][j] = 30
                else:
                    shepp_SD[i][j] = 30
        # convert image to qimage
        shepp_SD = qimage2ndarray.array2qimage(shepp_SD)
        return shepp_SD
'''