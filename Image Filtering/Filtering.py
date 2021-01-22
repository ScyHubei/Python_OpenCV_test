import cv2, os
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


def showbmp(imgfile):  # Canny
    img = cv2.imread(imgfile)
    print(img.shape)
    print(img.size)
    print(img.dtype)
    print(cv2.mean(img))

    cv2.imshow("img",img)
    imgviewx2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, imgviewx2 = cv2.threshold(imgviewx2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    title = '二值化'
    gbk_title = title.encode("gbk").decode(errors="ignore")

    cv2.imshow(gbk_title, imgviewx2)
    # img = cv2.GaussianBlur(img,(3,3),0)
    # cv2.imshow("GuassianBlur", img)
    canny = cv2.Canny(img,0,45)
    cv2.imshow("Canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sobel(imgfile):   # Sobel
    img = cv2.imread(imgfile)
    cv2.imshow("img", img)

    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absx = cv2.convertScaleAbs(x)
    absy = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    cv2.imshow("result", result)

    def getgray(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(img[y, x], result[y, x])

    cv2.setMouseCallback("img", getgray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Showarr(imgfile):  # 直方图
    img = np.array(Image.open(imgfile).convert('L'))

    plt.figure("lena")
    arr = img.flatten()
    n, bin, patches = plt.hist(arr, bins=256)
    plt.show()


def findContours(imgfile):
    image = cv2.imread(imgfile)
    BGR = np.array([100, 110, 120])
    upper = BGR + 30
    lower = BGR - 30
    mask = cv2.inRange(image, upper, lower)
    cv2.imshow("mask", mask)
    contoour, hicrarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contoour))
    allImage = image.copy()
    cv2.drawContours(allImage,contoour,-1,(0,0,255),2)
    cv2.imshow("Image of all",allImage)
    The = image.copy()
    contoour.sort(key=len, reverse=True)
    print(contoour)
    cv2.drawContours(The, [contoour[0],contoour[1]],-1,(0,0,255),2)
    cv2.imshow("1", The)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Laplacian(imageFile):
    img = cv2.imread(imageFile)
    # img = cv2.resize(img, None, fx=0.3, fy=0.3)
    Lap = cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(Lap)
    cv2.imshow("laplacian", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Filter(imgfile):
    img = cv2.imread(imgfile)
    imgSource = np.flip(img, axis=2)
    cv2.imshow("source", imgSource)
    #均值滤波
    img_mean = cv2.blur(img, (5, 5), 0)
    cv2.imshow("Mean", img_mean)
    # 高斯滤波
    img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow("Guassian", img_Guassian)
    # 中值滤波
    img_median = cv2.medianBlur(img, 5)
    cv2.imshow("median", img_median)
    # 双边滤波
    img_bilater = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imshow("bit", img_bilater)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    os.chdir(r'D:\Desktop\Image Filtering')
    # showbmp('3.bmp')
    # sobel("123.bmp")
    # Showarr("3.bmp")
    # findContours("2.jpg")
    # Laplacian("2.jpg")
    # Filter("2.jpg")
