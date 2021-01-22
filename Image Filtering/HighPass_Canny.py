import cv2
import numpy
from scipy import ndimage

  
def HighPass(): # 高通滤波
    Kernel_33 = numpy.array([[-1, -1, -1],
                             [-1, 0, -1],
                             [-1, -1, -1]])
    Kernel_55 = numpy.array([[-1, -1, -1, -1, -1],
                             [-1, 1, 2, 1, -1],
                             [-1, 2, 4, 2, -1],
                             [-1, 1, 2, 1, -1],
                             [-1, -1, -1, -1, -1]])
    img = cv2.imread("Gray.png", 0)
    k3 = ndimage.convolve(img, Kernel_33)
    K5 = ndimage.convolve(img, Kernel_55)

    blirred = cv2.GaussianBlur(img, (11,11), 0)
    g_hpf = img - blirred

    cv2.imshow("3*3", k3)
    cv2.imshow("5*5", K5)
    cv2.imshow("g_hpf", g_hpf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Canny():   #   Canny滤波
    img = cv2.imread("2.jpg")
    canny = cv2.Canny(img, 200, 300)
    cv2.imwrite("Canny.jpg", canny)
    cv2.imshow("img", img)
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def slice():
    img = numpy.zeros((200, 200), dtype=numpy.uint8)
    img[50:150, 50:150] = 255

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (0,255,0), 2)
    cv2.imshow("countours", color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # HighPass()
    # Canny()
    slice()
