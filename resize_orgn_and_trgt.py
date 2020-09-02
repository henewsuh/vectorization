# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:03:01 2020

@author: user
"""

from PIL import Image
import os
import cv2

original_dir='C:/Users/user/Desktop/resize/original/'
target_dir='C:/Users/user/Desktop/resize/target/'

resize_original_dir='C:/Users/user/Desktop/resize/resize_original/'
resize_target_dir= 'C:/Users/user/Desktop/resize/resize_target/'


def vis(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
def resize_original_image(original_dir):
    files = os.listdir(original_dir) 
    os.chdir(original_dir)
    for file in files:
        img = cv2.imread(file)
        resize_original_img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_AREA)
        os.chdir('../resize_original/')
        cv2.imwrite(file + '.png', resize_original_img)
        os.chdir("../original/")
    

def resize_target_image(target_dir):
    files = os.listdir(target_dir)
    os.chdir(target_dir)
    for file in files:
        img = cv2.imread(file)
        resize_target_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(file + '.png', resize_target_img)
        os.chdir('../resize_target/')
        cv2.imwrite(file + '.png', resize_target_img)
        os.chdir("../target/")
    

#===========================================================================================================#
        
resize_original_image(original_dir)
resize_target_image(target_dir)
    