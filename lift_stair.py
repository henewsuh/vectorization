import numpy as np

import os
import cv2
import geojson
import cv2 as cv
import scipy.spatial as spatial
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from osgeo import gdal
from osgeo import osr
from shapely.ops import linemerge
from skimage.morphology import thin
from skimage import img_as_ubyte
from shapely.geometry import shape


def raster_separation(filepath, fpnumber, separated_path): 
    
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]
    else:
        fpnumberr = fpnumber
    
    
    fpnumberpng = cv2.imread(filepath + '/' + fpnumberr + '.png')
    canvas1 = np.zeros(shape=fpnumberpng.shape, dtype=np.uint8)
    canvas1.fill(255)
    canvas2 = np.zeros(shape=fpnumberpng.shape, dtype=np.uint8)
    canvas2.fill(255)
    canvas3 = np.zeros(shape=fpnumberpng.shape, dtype=np.uint8)
    canvas3.fill(255)
    canvas4 = np.zeros(shape=fpnumberpng.shape, dtype=np.uint8)
    canvas4.fill(255)
    canvas5 = np.zeros(shape=fpnumberpng.shape, dtype=np.uint8)
    canvas5.fill(255)
    canvas6 = np.zeros(shape=fpnumberpng.shape, dtype=np.uint8)
    canvas6.fill(255)
    
    canvas1[np.where((fpnumberpng == [131,232,255]).all(axis = 2))] = [131,232,255] #wall
    canvas1[np.where((fpnumberpng == [0,255,255]).all(axis = 2))] = [0,255,255] #wall
    canvas2[np.where((fpnumberpng == [0,0,255]).all(axis = 2))] = [0,0,255] #window
    canvas3[np.where((fpnumberpng == [0,255,0]).all(axis = 2))] = [0,255,0] #door
    
    #for merged raster
    canvas4[np.where((fpnumberpng == [131,232,255]).all(axis = 2))] = [0,255,255] #wall
    canvas4[np.where((fpnumberpng == [0,255,255]).all(axis = 2))] = [0,255,255] #wall
    canvas4[np.where((fpnumberpng == [0,0,255]).all(axis = 2))] = [0,0,255] #window
    #for lift and stair
    canvas5[np.where((fpnumberpng == [161,89,131]).all(axis = 2))] = [161,89,131] #lift
    canvas6[np.where((fpnumberpng == [42,42,164]).all(axis = 2))] = [42,42,164] #stair
    canvas6[np.where((fpnumberpng == [62,89,175]).all(axis = 2))] = [42,42,164] #stair
    canvas6[np.where((fpnumberpng == [42,42,146]).all(axis = 2))] = [42,42,164] #stair
    
    wall_png = cv2.imwrite(separated_path + fpnumberr + '_wall.png', canvas1)
    window_png = cv2.imwrite(separated_path + fpnumberr + '_window.png', canvas2)
    door_png = cv2.imwrite(separated_path + fpnumberr + '_door.png', canvas3)
    merged_png = cv2.imwrite(separated_path + fpnumberr + '_merged.png', canvas4)
    lift_png = cv2.imwrite(separated_path + fpnumberr + '_lift.png', canvas5)
    stair_png = cv2.imwrite(separated_path + fpnumberr + '_stair.png', canvas6)
    os.chdir(filepath)
    
    return lift_png, stair_png

def vis(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def stair_box(filepath, outputpath, separated_path, fpnumber): 
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]
    else: 
        fpnumberr = fpnumber
        
    stair_img = cv2.imread(separated_path + '/' + fpnumberr + '_stair.png',0)
    ori_stair = stair_img.copy()
    stair = ~stair_img
    
    ret, stair_bin = cv.threshold(stair, 30, 255, 0)
    stair_contours, hierarchy = cv.findContours(stair_bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    stair_canvas1 = np.zeros(shape=stair_img.shape, dtype=np.uint8)
    stair_canvas1.fill(255)
    stair_canvas2 = np.zeros(shape=stair_img.shape, dtype=np.uint8)
    stair_canvas2.fill(255)
    
    cnt_pt = []
    for cnt in stair_contours:
        if cnt.size > 300 : #200 - 400 사이에서 조절
            gig_img = cv.drawContours(stair_img, [cnt], 0, (0, 0, 0), 3)  # blue   
            cnt_pt.append(cnt)
        else: #####조건문에 세부적으로 추가 필요 
            continue
    stair_cont = cv2.imwrite(outputpath + fpnumberr + '/' + fpnumberr + '_stair_cont.png', gig_img)
    
    #bounding box
    for i in range(len(cnt_pt)):
        dks = cnt_pt[i]
        rect = cv2.minAreaRect(dks)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bnd_img1 = cv2.drawContours(ori_stair,[box],0,(0,0,0),5)
        bnd_img2 = cv2.drawContours(stair_canvas2,[box],0,(0,0,0),5)
        
    stair_bnd1 = cv2.imwrite(filepath +  fpnumberr + '_stair_bnd1.png', bnd_img1)
    stair_bnd2 = cv2.imwrite(filepath + fpnumberr + '_stair_bnd2.png', bnd_img2)
    
    return stair_bnd2


def lift_box(filepath, outputpath, separated_path, fpnumber): 
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]
    else: 
        fpnumberr = fpnumber
        
    lift_img = cv2.imread(separated_path + '/' + fpnumberr + '_lift.png',0)
    ori_lift = lift_img.copy()
    lift = ~lift_img
    
    ret, lift_bin = cv.threshold(lift, 30, 255, 0)
    lift_contours, hierarchy = cv.findContours(lift_bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    lift_canvas1 = np.zeros(shape=lift_img.shape, dtype=np.uint8)
    lift_canvas1.fill(255)
    lift_canvas2 = np.zeros(shape=lift_img.shape, dtype=np.uint8)
    lift_canvas2.fill(255)
    
    cnt_pt = []
    for cnt in lift_contours:
        if cnt.size > 400 : #200 - 400 사이에서 조절
            gig_img = cv.drawContours(lift_img, [cnt], 0, (0, 0, 0), 3)  # blue   
            cnt_pt.append(cnt)
        else: #####조건문에 세부적으로 추가 필요 
            continue
    lift_cont = cv2.imwrite(outputpath + fpnumberr + '/' + fpnumberr + '_lift_cont.png', gig_img)
    
    #bounding box
    for i in range(len(cnt_pt)):
        dks = cnt_pt[i]
        rect = cv2.minAreaRect(dks)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        bnd_img1 = cv2.drawContours(ori_lift,[box],0,(0,0,0),5)
        bnd_img2 = cv2.drawContours(lift_canvas2,[box],0,(0,0,0),5)
        
    lift_bnd1 = cv2.imwrite(filepath +  fpnumberr + '_lift_bnd1.png', bnd_img1)
    lift_bnd2 = cv2.imwrite(filepath + fpnumberr + '_lift_bnd2.png', bnd_img2)
    
    return lift_bnd2

def contrast_binarization(filepath, fpnumber, outputpath, src): 
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]  
    else: 
        fpnumberr = fpnumber 
    
    img = cv2.imread(filepath + fpnumberr + '_' + src + '_thin.bmp')
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)  # 이진화 시행
    contrast = ~binary #흑백반전
    
    cv2.imwrite(filepath + fpnumberr + '_' + src + '_bin.bmp', contrast)


def line_generation(filepath, fpnumber, src):
    
    from raster2vec import raster2vec as r2v
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]  
    else: 
        fpnumberr = fpnumber 
        
    file_name = str(fpnumberr) + '_' + src + '_bin.bmp'
    srccc = '_' + src + '_bin.bmp'
    
    line_vec = r2v(filepath, file_name, fpnumberr, srccc)
    line_vec = [shape(i['geometry']) for i in line_vec['features']]
    
    
#=================================================================================================================================================

    
mainpath = 'C:/Users/user/Desktop/module3_vec/module3/'   
filepath = 'C:/Users/user/Desktop/real/'
outputpath = 'C:/Users/user/Desktop/module3_vec/module3/test_output/'
separated_path = 'C:/Users/user/Desktop/module3_vec/module3/separated_output/'
fpnumber = 't900-3.png'
fpnumberr = 't900-3'


raster_separation(filepath, fpnumber, separated_path)
stair_bb = stair_box(filepath, outputpath, separated_path, fpnumber)
lift_bb = lift_box(filepath, outputpath, separated_path, fpnumber)

stair_contbin = contrast_binarization(filepath, fpnumber, outputpath, src="stair")
os.chdir(mainpath)
from raster2vec import raster2vec as r2v
stair_vec = line_generation(filepath, fpnumber, src="stair")















