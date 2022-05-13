# -*- coding = utf-8 -*-
# @File_name = image_preprocessing.py
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def highPassFilter(image, d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    fimg = np.log(np.abs(fshift))

    # 高通滤波器的实现函数的定义
    def make_transform_matrix(d):
        transfor_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x: (x - 1) / 2, fimg.shape))
        for i in range(transfor_matrix.shape[0]):
            for j in range(transfor_matrix.shape[1]):
                def cal_distance(pa, pb):
                    from math import sqrt
                    dis = sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    return dis

                dis = cal_distance(center_point, (i, j))
                if dis <= d:
                    transfor_matrix[i, j] = 0
                else:
                    transfor_matrix[i, j] = 1
        return transfor_matrix

    d_matrix = make_transform_matrix(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * d_matrix)))
    return new_img


if __name__ == "__main__":
    img = cv.imread('COVID-19_Radiography_Dataset/Normal/images/Normal-1.png', 0)
    img = highPassFilter(img, 8)
    plt.imshow(img, cmap="gray")
    plt.show()
