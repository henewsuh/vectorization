# -*- coding: utf-8 -*-
"""
harrison corner detection parameter test 
"""
import numpy as np
import cv2
import os

print(os.getcwd())
wall_filename = '7-4wall.png'
window_filename = '80-1window.png'
door_filename = '_door.bmp'

wall_img = cv2.imread(wall_filename, 0)
window_img = cv2.imread(window_filename, 0)
door_img = cv2.imread(door_filename, 0)

height, width = wall_img.shape

#find harris corner detection for wall
print("wall harris corner starts!")
wall_dst = cv2.cornerHarris(wall_img, 2, 3, 0.04)
wall_dst = cv2.dilate(wall_dst,None)
ret, wall_dst = cv2.threshold(wall_dst, 0.01*wall_dst.max(), 255, 0)
wall_dst = np.uint8(wall_dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(wall_dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(wall_img, np.float32(centroids), (5,5), (-1,-1), criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
res = res[1:len(res),:]
#blank = np.zeros((512,512) , dtype = int)
corner_harris = []
for i,p in enumerate(res):
    x,y = p[2:4]
    corner_harris.append([i,x,height-y])
    cv2.circle(wall_img, (x,y), 4 , (0,255,0) , 2)
cv2.imwrite('wallcorner.png' , wall_img)
print("wall harris corner ends!")



##find harris corner detection for window
#print("wall harris corner starts!")
#window_dst = cv2.cornerHarris(window_img, 2, 3, 0.04)
#window_dst = cv2.dilate(window_dst,None)
#ret, window_dst = cv2.threshold(window_dst, 0.01*window_dst.max(), 255, 0)
#window_dst = np.uint8(window_dst)
#ret, labels, stats, centroids = cv2.connectedComponentsWithStats(window_dst)
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#corners = cv2.cornerSubPix(window_img, np.float32(centroids), (5,5), (-1,-1), criteria)
#
## Now draw them
#res = np.hstack((centroids,corners))
#res = np.int0(res)
#res = res[1:len(res),:]
##blank = np.zeros((512,512) , dtype = int)
#corner_harris = []
#for i,p in enumerate(res):
#    x,y = p[2:4]
#    corner_harris.append([i,x,height-y])
#    cv2.circle(window_img, (x,y), 4 , (0,255,0) , 2)
#cv2.imwrite('windowcorner.png' , window_img)
#print("window harris corner ends!")