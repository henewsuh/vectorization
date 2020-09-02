# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:22:33 2020

@author: user
"""
import cv2
import os
import numpy as np


overlay_filepath = 'C:/Users/user/Desktop/ssss'
background_filepath = 'C:/Users/user/Desktop/ss2'

os.chdir(overlay_filepath)
files = os.listdir() 


overlay = cv2.imread(overlay_filepath + '/' + '10-1.bmp')
wall_background = cv2.imread(background_filepath + '/' + '10-1wall.png')
window_background = cv2.imread(background_filepath + '/' + '10-1window.png')

aa = overlay + wall_background
bb = overlay + window_background 

canvas_wall = np.zeros(shape=overlay.shape, dtype=np.uint8)
canvas_wall.fill(255)
canvas_wall[np.where((aa == [131,232,255]).all(axis = 2))] = [0,0,0] #wall
cv2.imwrite('10-1_overlay_wall.png' , canvas_wall)


canvas_window = np.zeros(shape=overlay.shape, dtype=np.uint8)
canvas_window.fill(255)
canvas_window[np.where((bb == [0,0,255]).all(axis = 2))] = [0,0,0] #
cv2.imwrite('10-1_overlay_window.png' , canvas_window)




#window_background = cv2.imread(background_filepath + '/' + '10-1window.png')
    


def vis(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

