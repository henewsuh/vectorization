# -*- coding: utf-8 -*-
"""
window coordinate extraction
200721 
"""
import cv2
import os
import numpy as np



def vis(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def window_nodes_extraction(filepath, separated_path , fpnumber, outputpath):
    
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]    
    
    every_sk = cv2.imread(outputpath + '/' + fpnumberr + '.png/' + fpnumberr + '_thin.bmp')    
    window_sk = cv2.imread(outputpath + '/' + fpnumberr + '.png/' + fpnumberr + '_window_thin.bmp')        
    window_background = cv2.imread(separated_path + '/' + fpnumberr + '_window.png')
    
    window = every_sk + window_background 
    
    canvas_window = np.zeros(shape=every_sk.shape, dtype=np.uint8)
    canvas_window.fill(255)
    canvas_window[np.where((window == [0,0,255]).all(axis = 2))] = [0,0,0]
    
#    color = (0, 0, 0)
#    a = np.argwhere(canvas_window == color)
#    window_coord = a[::3][:, [0, 1]]
#    print(window_coord)
#    print(window_coord.shape)
    
    colour = [0, 0, 0]
    x, y = np.where(np.all(canvas_window == colour, axis = 2))
    window_coordd = np.c_[x, y]
    print(window_coordd)
    print(window_coordd.shape)
    
    

filepath = 'C:/Users/user/Desktop/module3_vec/module3/test_input'
outputpath = 'C:/Users/user/Desktop/module3_vec/module3/test_output'
separated_path = 'C:/Users/user/Desktop/module3_vec/module3/separated_output'
fpnumber = '1-1.png' 

window_nodes_extraction(filepath, separated_path, fpnumber, outputpath)

