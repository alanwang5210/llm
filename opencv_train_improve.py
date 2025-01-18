import cv2
import numpy as np

#灰度图片
img = cv2.imread('C:\\Users\\10100\\Downloads\\a.jpg', 0)
#直方图均衡化操作。这个方法的输入是一幅灰度图像，输出则是经过直方图均衡化处理的图像
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
cv2.imshow('img', res)

#CLAHE有限对比适应性直方图均衡化方法的处理效果
clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
cl1 = clahe.apply(img)
cv2.imwrite('clahe_2.jpg',cl1)

img = cv2.imread('C:\\Users\\10100\\Downloads\\a.jpg')
#对彩色图像进行直方图均衡化
(b,g,r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH,gH,rH),)
res = np.hstack((img,result))
cv2.imshow('dst',res)
cv2.waitKey(0)