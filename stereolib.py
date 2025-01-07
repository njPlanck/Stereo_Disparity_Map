import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
imgL = cv.imread('view_l.png', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('view_r.png', cv.IMREAD_GRAYSCALE)
'''
plt.imshow(imgL)
plt.show()
plt.imshow(imgL)
plt.show()
''' 
stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
disparity_normalised = cv.normalize(disparity,None,alpha=0,beta=255,norm_type=cv.NORM_MINMAX,dtype=cv.CV_8U)
fig,ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(disparity_normalised,"gray")
ax[1].imshow(disparity,"gray")
plt.tight_layout()
plt.show()