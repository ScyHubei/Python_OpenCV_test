import cv2
import sys
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication
from PyQt5.QtGui import QPixmap, QImage
from Qt import Ui_MainWindow
from PyQt5.QtCore import Qt
import numpy as np
from functools import partial


def Qlabelimg(img):
    a = QImage(img.data2, img.width, img.height, img.width*img.bands, QImage.Format_RGB888)
    ui.LabelImage_2.setPixmap(QPixmap.fromImage(a))
    ui.LabelImage_2.setAlignment(Qt.AlignCenter)


def path():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    imgpath = filedialog.askopenfilename()
    return imgpath


class labelimage():
    def __init__(self):
        self.data = None
        self.height = None
        self.width = None
        self.bands = None
        self.path = None
        self.data2 = None

    # File #
    def open(self):
        try:
            self.path = path()
            self.data = cv2.imread(self.path)
            self.height = self.data.shape[0]
            self.width = self.data.shape[1]
            self.bands = self.data.shape[2]
            self.data = cv2.cvtColor(img.data, cv2.COLOR_BGR2RGB)
            a = QImage(self.data, self.width, self.height, self.width * self.bands, QImage.Format_RGB888)
            ui.LabelImage.setPixmap(QPixmap.fromImage(a))
            ui.LabelImage.setAlignment(Qt.AlignCenter)
        except:
            print("没有选择图片或格式错误")

    def savejpg(self):
        temp = cv2.cvtColor(self.data2, cv2.COLOR_RGB2BGR)
        cv2.imwrite("Save_image.jpg", temp)
        del temp

    # Filter #
    def mean(self):
        self.data2 = cv2.blur(self.data, (5, 5))
        Qlabelimg(img)

    def median(self):
        self.data2 = cv2.medianBlur(self.data, 5)
        Qlabelimg(img)

    def gaussian(self):
        self.data2 = cv2.GaussianBlur(self.data, (5, 5), 0)
        Qlabelimg(img)

    def laplacian(self):
        self.data2 = cv2.Laplacian(self.data, cv2.CV_8U)
        Qlabelimg(img)

    def scharr(self):
        self.data2 = cv2.Scharr(self.data, cv2.CV_8U, 1, 0)
        Qlabelimg(img)

    def sobel(self):
        sobelx = cv2.Sobel(self.data, cv2.CV_8U, 1, 0, ksize=7)
        sobely = cv2.Sobel(self.data, cv2.CV_8U, 0, 1, ksize=7)
        self.data2 = sobely + sobelx
        Qlabelimg(img)
        del sobelx, sobely

    def canny(self):
        self.data2 = cv2.Canny(self.data, threshold1=100, threshold2=200)
        self.data2 = cv2.cvtColor(self.data2, cv2.COLOR_GRAY2RGB)
        Qlabelimg(img)

    # Morphology #
    def dilate(self):
        kernel = np.ones((5, 5), np.uint8)
        self.data2 = cv2.dilate(self.data, kernel, iterations=1)
        Qlabelimg(img)
        del kernel

    def erosion(self):
        kernel = np.ones((5, 5), np.uint8)
        self.data2 = cv2.erode(self.data, kernel, iterations=1)
        Qlabelimg(img)
        del kernel

    def open_operation(self):
        kernel = np.ones((5, 5), np.uint8)
        self.data2 = cv2.morphologyEx(self.data, cv2.MORPH_OPEN, kernel)
        Qlabelimg(img)
        del kernel

    def close_operation(self):
        kernel = np.ones((5, 5), np.uint8)
        self.data2 = cv2.morphologyEx(self.data, cv2.MORPH_CLOSE, kernel)
        Qlabelimg(img)
        del kernel


def button():
    # File #
    ui.actionopen.triggered.connect(img.open)
    ui.actionsave.triggered.connect(img.savejpg)

    # Filter #
    ui.actionMean.triggered.connect(img.mean)
    ui.actionMedian.triggered.connect(img.median)
    ui.actionGaussianBlur.triggered.connect(img.gaussian)
    ui.actionScharr.triggered.connect(img.scharr)
    ui.actionSobel.triggered.connect(img.sobel)
    ui.actionLaplace.triggered.connect(img.laplacian)
    ui.actionCanny.triggered.connect(img.canny)

    # Morphology #
    ui.actionDilate.triggered.connect(img.dilate)
    ui.actionErosion.triggered.connect(img.erosion)
    ui.actionOpen_operation.triggered.connect(img.open_operation)
    ui.actionClose_operation.triggered.connect(img.close_operation)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    img = labelimage()
    button()

    MainWindow.show()
    sys.exit(app.exec_())