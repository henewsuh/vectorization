# -*- coding: utf-8 -*-
"""
thinning algorithm in opencv
for temp test
"""
import os
import cv2
from skimage import img_as_bool, io, color, morphology
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread("80-1window.png")
thinned = cv2.ximgproc.thinning(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
cv2.imwrite('80-window-cv2-thinned.png', thinned)


image2 = img_as_bool(color.rgb2gray(cv2.imread("80-1wall.png")))
thinned2 = morphology.medial_axis(image2)
thinned2 = thinned2 * 255
cv2.imwrite('80-wall-skimage-thinned.png', thinned2)

image3 = img_as_bool(color.rgb2gray(cv2.imread("80-1window.png")))
thinned3 = morphology.medial_axis(image3)
thinned3 = thinned3 * 255
cv2.imwrite('80-window-skimage-thinned.png', thinned3)

image4 = cv2.imread("aa.jpg")
thinned4 = cv2.ximgproc.thinning(cv2.cvtColor(image4, cv2.COLOR_RGB2GRAY))
cv2.imwrite('8aa.jpg', thinned4)



img = cv2.imread("900-3wall.png", 0)

cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
rec = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
rec2 = rec = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

dilation = cv2.dilate(img, rec, iterations = 1)
erosion = cv2.erode(dilation, rec ,iterations = 1)
dilation2 = cv2.dilate(erosion, rec2, iterations = 1)
cv2.imwrite('900-3wall_mt2.png', dilation2)

#cv2.imwrite('900-3door_mt.png', dilation)

