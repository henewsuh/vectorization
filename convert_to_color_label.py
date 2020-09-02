# -*- coding: utf-8 -*-
"""
각자 출력된 벽, 문, 등의 3차년도 결과를 우리의color label형태로 변환하는 코드
"""

import os
import cv2 
import numpy as np

dir = "C:/Users/user/Desktop/3차년도 레스터 테스트 결과/"
os.chdir(dir)
bldg = os.listdir()

def vis (image): 
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
for folder in bldg:
    if folder == 'merged_output':
        continue 
    
    os.chdir("./" + folder)
    
    
    files = os.listdir() 
    for file in files:
        if 'png' in file and 'door' in file:
            door_png = ~(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
        if 'png' in file and 'wall' in file:
            wall_png = ~(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
        if 'png' in file and 'window' in file:
            window_png = ~(cv2.imread(file, cv2.IMREAD_GRAYSCALE))
     
        
    #binarization
    _, bi_door_png = cv2.threshold(door_png, thresh=5, maxval=255, type=cv2.THRESH_BINARY)
    bi_door_png = ~bi_door_png
    bi_wall_png = ~(cv2.threshold(wall_png, thresh=5, maxval=255, type=cv2.THRESH_BINARY)[1])                             
    bi_window_png = ~(cv2.threshold(window_png, thresh=5, maxval=255, type=cv2.THRESH_BINARY)[1])
    
    
    #GRAYtoBGR
    door_png = cv2.cvtColor(bi_door_png, cv2.COLOR_GRAY2BGR)
    wall_png = cv2.cvtColor(bi_wall_png, cv2.COLOR_GRAY2BGR)
    window_png = cv2.cvtColor(bi_window_png, cv2.COLOR_GRAY2BGR)
    
    
    #change specific pixel color
    door_png[np.where((door_png == [0,0,0]).all(axis = 2))] = [0,255,0]
    wall_png[np.where((wall_png == [0,0,0]).all(axis = 2))] = [0,255,255]
    window_png[np.where((window_png == [0,0,0]).all(axis = 2))] = [0,0,255]        


    #merge
    canvas = door_png.copy()    
    canvas[np.where((wall_png == [0,255,255]).all(axis = 2))] = [0,255,255]
    canvas[np.where((window_png == [0,0,255]).all(axis = 2))] = [0,0,255]
    canvas[np.where((canvas == [255,255,255]).all(axis = 2))] = [255,0,0]
    
    
    os.chdir('../merged_output/')
    cv2.imwrite(folder + '.png', canvas)
  
    os.chdir("../")
    




