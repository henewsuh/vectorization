# -*- coding: utf-8 -*-
"""
door corner detection test 
200616 
"""

import numpy as np
import cv2
import os
from PIL import Image


def vis(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def countblack(px,py,img,size):
    h1 = int(size/2)
    h2 = size - h1
    rect = img[py-h1:py+h2,px-h1:px+h2]
    dot = rect[rect == 0 ]
    return float(len(dot))/(size*size)


def good_feature_corner_extraction(filepath, fpnumber):
    #import image
    filename = filepath + '/' + fpnumber 
    dong = int(''.join(filter(str.isdigit, str(fpnumber))))
    img = cv2.imread(filename, 0)
    #image size
    height, width = img.shape
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)  #blur image
    img_th2 = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2) #binarization image
    original = img.copy()
    
    corners = cv2.goodFeaturesToTrack(img, 500, 0.01, 20)
    corners = np.int0(corners)
    
    corner_gdft = []
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3 , 0 , 2)
    cv2.imwrite('gdf' + str(dong) + '_corner.png' , img)
    corner_gdft = np.asarray(corner_gdft)    
        
        

def door_corner_extraction(filepath , fpnumber):
    #import image
    filename = filepath + '/' + fpnumber 
    dong = int(''.join(filter(str.isdigit, str(fpnumber))))
    img = cv2.imread(filename,0)
    #image size
    height, width = img.shape
    original = img.copy()
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    #Harris corner detection
    dst = cv2.cornerHarris(img,3,7,0.05)
    

    #binarization of image to do Connected Component Analysis
    ret, Ithresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # CCA(label, stats, area)
    comp = cv2.connectedComponentsWithStats(Ithresh)
    #label 
    door_label = comp[1]
    
    #label extenstion for 4pixels to every direction
    new_label = np.full((height, width), 255) #blank image
    for row in range(3,height-3):
        for col in range(3,width-3):
            if door_label[row,col] != 0:
                new_label[row-3:row+3,col-3:col+3] = door_label[row,col] #extended image
        

    #result is dilated for marking the corners, not important
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    #Simplification
    #CCA to every cluster of corners
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria) #centroid of dst

    # candidate of points from door
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    res = res[1:len(res),:]

    #extraction complete

    #extract only 1pixel(1 coordinate) for every corner
    black = []
    ind = []
    for p in res:
        black.append(countblack(p[2],p[3],img,9))
    
    #threshold = 0.1 to check if the corner is valid
    for i, r in enumerate(black):
        if r < 0.1:
            ind.append(i)

    for i in reversed(ind):
        res = np.delete(res,i,0)
    
    #save coordinates
    corner_harris = []
    for i in res:
        y,x = i[2:4]
        label = new_label[x,y]
        corner_harris.append([label,y,width-x])
        cv2.circle(img,(y,x), 3 , 0 , 2) 
        #cv2.circle(img, (center), radian, color, thickness)
    cv2.imwrite(str(dong) + 'doorcorner.jpg' , img)
    corner_harris = np.asarray(corner_harris)

    return corner_harris


# ========================================================================================================================

def contour(filepath, fpnumber): 
    filename = filepath + '/' + fpnumber 
    dong = int(''.join(filter(str.isdigit, str(fpnumber))))
    img = cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    test = binary

#    bin_img = Image.fromarray(binary)
#    cv2.imwrite('bin_' + str(dong) + '.png', test)
    
    contours, h = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    #cv2.imwrite('bin_' + str(dong) + '.png', image)

    canvas = np.zeros(shape=img.shape, dtype=np.uint8)
    canvas.fill(255)
    canvas[np.where((image == [0,0,0]).all(axis = 2))] = [255,255,255]
    canvas[np.where((image == [0,255,0]).all(axis = 2))] = [0,0,0]
    cv2.imwrite('cntr' + str(dong) + '.png', canvas)
    
# ========================================================================================================================


filepath = 'C:/Users/user/Desktop/original_door'

print("start!")
print(os.getcwd())
os.chdir(filepath)
files = os.listdir() 

#for file in files:
#    print(file + ' start!')
#    #good_feature_corner_extraction(filepath, file)
#    contour(filepath, file)
#
#os.chdir(filepath)

for file in files: 
    print("Harris CD starts: " + file)
    door_corner_extraction(filepath, file)