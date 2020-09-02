# -*- coding: utf-8 -*-
"""
merge door and window raster output from DL
merged raster --> skeletonization --> classification using overlay
"""

import os
import cv2 
import numpy as np

filepath = 'C:/Users/user/Desktop/raster_output'
outputpath = 'C:/Users/user/Desktop/label_merged_ww_output'


print("start!")
print(os.getcwd())
os.chdir(filepath)
print(os.getcwd())
files = os.listdir() 



for file in files:
    
    cur_path = os.path.basename(file)
    fpnumber = os.path.splitext(cur_path)[0]
    
    fpnumberpng = cv2.imread(filepath + '/' + cur_path)
    canvas1 = np.zeros(shape=fpnumberpng.shape, dtype=np.uint8)
    canvas1.fill(255)

    
    canvas1[np.where((fpnumberpng == [131,232,255]).all(axis = 2))] = [131,232,255] #wall
    canvas1[np.where((fpnumberpng == [0,255,255]).all(axis = 2))] = [0,255,255] #wall
    canvas1[np.where((fpnumberpng == [0,0,255]).all(axis = 2))] = [0,0,255] #window

    os.chdir(outputpath)
    
    #save as rgb
    merged_ww = cv2.imwrite(fpnumber + 'label_merged_ww.png', canvas1)
   
#    #save as grayscale 
#    merged_ww = cv2.cvtColor(canvas1, cv2.COLOR_BGR2GRAY)
#    merged_ww = cv2.imwrite(fpnumber + 'merged_ww.png', merged_ww)
#    
    
    
    
    
    