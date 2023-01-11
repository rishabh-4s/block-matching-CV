import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imgL = cv.imread('tsukuba_l.png', 0)
imgR = cv.imread('tsukuba_r.png',0)
# imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
# imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
stereo = cv.StereoSGBM_create(numDisparities=16, blockSize=11)
disparity = stereo.compute(imgL,imgR)
# plt.imshow(disparity)
plt.imshow(disparity,'gray')
plt.axis('off')
plt.show()
