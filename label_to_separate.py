# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 17:27:22 2020

@author: user
"""
import os
import numpy as np
import cv2

filepath = 'C:/Users/user/Desktop/raster_output'
outputpath = 'C:/Users/user/Desktop/label_separated_output'
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
    canvas2 = np.zeros(shape=fpnumberpng.shape, dtype=np.uint8)
    canvas2.fill(255)
    canvas3 = np.zeros(shape=fpnumberpng.shape, dtype=np.uint8)
    canvas3.fill(255)
    
    canvas1[np.where((fpnumberpng == [131,232,255]).all(axis = 2))] = [131,232,255] #wall
    canvas2[np.where((fpnumberpng == [0,0,255]).all(axis = 2))] = [0,0,255] #window
    canvas3[np.where((fpnumberpng == [0,255,0]).all(axis = 2))] = [0,255,0] #door
    
    os.chdir(outputpath)
    
    wall_png = cv2.imwrite(fpnumber + 'wall.png', canvas1)
    window_png = cv2.imwrite(fpnumber + 'window.png', canvas2)
    door_png = cv2.imwrite(fpnumber + 'door.png', canvas3)
    