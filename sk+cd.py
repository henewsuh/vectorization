# -*- coding: utf-8 -*-
"""
raster corner detection + sk 
"""

import numpy as np
import cv2
import os



filepath = 'C:/Users/user/Desktop'
os.chdir(filepath)
print(os.getcwd())
wall_sk = cv2.imread(filepath + '/' + '10-1_overlay_wall.png', 0)
wall_png = cv2.imread(filepath + '/' + '10-1wall.png', 0)



height, width = wall_png.shape

#find harris corner detection for wall
print("wall harris corner starts!")
wall_dst = cv2.cornerHarris(wall_png, 21, 21, 0.04)
wall_dst = cv2.dilate(wall_dst,None)
ret, wall_dst = cv2.threshold(wall_dst, 0.01*wall_dst.max(), 255, 0)
wall_dst = np.uint8(wall_dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(wall_dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(wall_png, np.float32(centroids), (5,5), (-1,-1), criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
res = res[1:len(res),:]
#blank = np.zeros((512,512) , dtype = int)
corner_harris = []
for i,p in enumerate(res):
    x,y = p[2:4]
    corner_harris.append([i,x,height-y])
    cv2.circle(wall_sk, (x,y), 4 , (0,255,0) , 2)
cv2.imwrite('wallcorner.png' , wall_sk)
print("wall harris corner ends!")



