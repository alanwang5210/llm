import cv2
import numpy as np
#图像去噪

# 均值滤波（适合去除随机噪声）
def blur_demo(image):
    dst = cv2.blur(image, (1, 15))
    cv2.imshow("blur", dst)


# 中值滤波（适合去除椒盐噪声）
def median_blur_demo(image):
    dst = cv2.medianBlur(image, 5)
    cv2.imshow("median_blur", dst)


# 将图像与内核进行卷积，将任意线性滤波器应用于图像
def custom_blur_demo(image):
    kernel = np.ones([5, 5], np.float32) / 25
    dst = cv2.filter2D(image, -1, kernel=kernel)
    cv2.imshow("custom_blur", dst)


# 高斯双边滤波
def bi_demo(image):
    dst = cv2.bilateralFilter(image, 0, 100, 5)
    cv2.imshow("bi_demo", dst)


# 均值迁移滤波
def shift_demo(image):
    dst = cv2.pyrMeanShiftFiltering(image, 10, 50)
    cv2.imshow("shift_demo", dst)


def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv


# 为图像增加高斯噪声
def gaussian_noise(image):
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv2.imshow("noise image", image)


src = cv2.imread("C:\\Users\\10100\\Downloads\\a.jpg")
img = cv2.resize(src, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
cv2.imshow('input_image', img)
blur_demo(img)
median_blur_demo(img)
custom_blur_demo(img)

bi_demo(img)
shift_demo(img)

gaussian_noise(src)
# 通过cv2库提供的GaussianBlur()方法去除高斯噪声
dst = cv2.GaussianBlur(src, (15, 15), 0)
cv2.imshow("Gaussian_Blur2", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
