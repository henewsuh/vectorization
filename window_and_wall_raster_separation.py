
"""
merged label (wall, door, window) to separated label (wall, window)
"""
import os
import numpy as np
import cv2

filepath = 'C:/Users/user/Desktop/raster_output'
outputpath = 'C:/Users/user/Desktop/ww_separated'
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
    
    canvas1[np.where((fpnumberpng == [131,232,255]).all(axis = 2))] = [0,255,255] #wall
    canvas1[np.where((fpnumberpng == [0,255,0]).all(axis = 2))] = [0,0,255] #window
    
    os.chdir(outputpath)
    
    wall_png = cv2.imwrite(fpnumber + '_merged.png', canvas1)