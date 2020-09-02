import numpy as np
from PIL import Image
import os
import geojson as gj
import cv2
import networkx as nx
import geopandas as gpd
from shapely.geometry import shape
import geojson
from shapely.geometry import Point as point
from shapely.geometry import LineString as string
from shapely.affinity import affine_transform as af
from shapely.geometry import Polygon, mapping
from geojson import Feature, FeatureCollection, dump
import shapely as sp
from osgeo import gdal
from osgeo import osr
from shapely.ops import linemerge
from skimage.morphology import thin
from skimage import img_as_ubyte
import scipy.spatial as spatial
from shapely.geometry import MultiLineString
from shapely.ops import polygonize
import fiona
from shapely.ops import split
import shutil
import json
import time

def bi_check(separated_path, fpnumber, fpnumberr, src):
    
    img = cv2.imread(separated_path + fpnumber + src + '.png')
    height = img.shape[0]
    width = img.shape[1]
    
    find = [255, 255, 255]
    result = np.count_nonzero(np.all(img == find,axis=2))
    total = height * width 
    
    fake = total - result 
    
    bi_ratio = (fake/total) * 100
    
    return bi_ratio



def empty_check(fpnumber, fpnumberr, outputpath, src):
    
    img = cv2.imread(outputpath + fpnumber + '/' + fpnumber + src + '_thin.bmp', 0)
    ck = ~img 
    zero = ck.sum() 
    
    return zero 

def bi_thinning(fpnumber, fpnumberr, outputpath, separated_path, src):   
    
    img_name = fpnumber + '/' + fpnumber + src + '_bnd2.png'
    img = cv2.imread(outputpath + img_name)
    
    try:
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    except:
        return None
    
    ret, image = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    BW_Original = image/255

    def pre_thinning1(image):
        rows, columns = image.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P1,P2,P3,P4,P5,P6,P7,P8 = n = neighbours(x, y, image)
                B_odd = P1 + P3 + P5 + P7
                if B_odd < 2:
                    image[y][x] = 0
                elif B_odd > 2:
                    image[y][x] = 1
                else:
                    image[y][x] = image[y][x]
        return image

    def neighbours(x,y,image):
        """Return 8-neighbours of image point P(x,y), in a clockwise order
        P4 P3 P2
        P5  P  P1
        P6 P7 P8
        """
        img = image
        x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
        plist = [ img[y][x1], img[y_1][x1], img[y_1][x], img[y_1][x_1], img[y][x_1], img[y1][x_1], img[y1][x], img[y1][x1] ] #P1 ~ P8
        return plist

    def transitions(neighbours):
        "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
        n = neighbours + neighbours[0:1]      # P1, P2, ... , P7, P8, P1
        return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P1,P2), (P2,P3), ... , (P7,P8), (P8,P1)

    def modified(image):
        "modified Thinning Algorithm"
        Image_Thinned = image.copy()  # deepcopy to protect the original image
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        changing1 = changing2 = 1        #  the points to be removed (set as 0)
        count = 0
        one_list = []
        tmp_list = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                if Image_Thinned[y][x] == 1:
                    one_list.append((x, y))

        while changing1 or changing2:   #  iterates until no further changes occur in the image
            # Step 1
            changing1 = []

            for x, y in one_list:
                P1, P2, P3, P4, P5, P6, P7, P8 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[y][x] == 1 and    # Condition 0: Point P in the object regions
                    2 <= sum(n) <= 6 and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and # Condition 2:
                    P1 * P3 * P7 == 0 and   # Condition 3
                    P1 * P5 * P7 == 0):     # Condition 4
                    changing1.append((x, y))
                else:
                    tmp_list.append((x, y))

            for x, y in changing1:
                Image_Thinned[y][x] = 0

            one_list = tmp_list
            tmp_list = []

            # Step 2
            changing2 = []

            for x, y in one_list:
                P1, P2, P3, P4, P5, P6, P7, P8 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[y][x] == 1     and    # Condition 0: Point P in the object regions
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2:
                    P1 * P5 * P7 == 0  and    # Condition 3
                    P3 * P5 * P7 == 0):         # Condition 4
                    changing2.append((x, y))
                else:
                    tmp_list.append((x, y))

            one_list = tmp_list
            tmp_list = []

            for x, y in changing2:
                Image_Thinned[y][x] = 0

            count = count + 1
            if count > 20:
                break
        return Image_Thinned
    "Apply the algorithm on images"
    BW_Skeleton = 255 - modified(pre_thinning1(BW_Original))*255
     
    cv2.imwrite(outputpath + fpnumber + '/' + fpnumber + src + '_thin.bmp' , BW_Skeleton)
    return BW_Skeleton


def door_corner_extraction(outputpath , fpnumber, fpnumberr):

    #import image
    filename = outputpath + fpnumber + '/' + fpnumber + '_door.bmp' 
    img = cv2.imread(filename, 0)
    #image size
    height, width = img.shape
    original = img.copy()

    
    #Harris corner detection
    dst = cv2.cornerHarris(img,5,3,0.05) #3 or 5

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
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    #Simplification
    #CCA to every cluster of corners
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria)

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
        corner_harris.append([label, y, width-x])
        cv2.circle(img,(y,x),3 , 0 , 2)
    
    if fpnumber.endswith('.png'):
        fpnumber = fpnumber[:-4]  
    cv2.imwrite(outputpath + fpnumber + '/'  + fpnumber + '_door_corner.png' , img)
    corner_harris = np.asarray(corner_harris)

    return corner_harris


def harris_tolist(harris_corner):
    corner = harris_corner
    clist = corner.tolist()
    ccoord = [] 
    for i in range(len(clist)):
        x = clist[i][1]
        y = clist[i][2]
        ccoord.append([x,y]) 
        
    return ccoord


def countblack(px,py,img,size):
    h1 = int(size/2)
    h2 = size - h1
    rect = img[py-h1:py+h2,px-h1:px+h2]
    dot = rect[rect == 0 ]
    return float(len(dot))/(size*size)


def wall_corner_nearest(outputpath, fpnumber):
    def wall_corner_extraction_harris(img):
        
        if fpnumber.endswith('.png'):
            fpnumberr = fpnumber[:-4]    
        
        height, width = img.shape
        original = img.copy()
        
        dst = cv2.cornerHarris(img,7,3,0.04)
        
        dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria)
        
        # Now draw them
        res = np.hstack((centroids,corners))
        res = np.int0(res)
        res = res[1:len(res),:]
        #blank = np.zeros((512,512) , dtype = int)
        real_corners = []
        corner_harris = []
        for i,p in enumerate(res):
            x,y = p[2:4]
            real_corners.append((x,y))
            corner_harris.append([i,x,height-y])
            cv2.circle(original,(x,y),3 , 0 , 2) #dump the point(in the form of circle) into the copy of the original image
            
        cv2.imwrite(outputpath + fpnumberr + '/' + fpnumberr + '_wall_corner.png' , original)
        return real_corners # [id of node , x, y]
    
    
    outputpath = outputpath
    fpnumber = fpnumber 
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]   
    
    test_img = cv2.imread(outputpath + '/' + fpnumberr + '.png/' + fpnumberr + '_wall_thin.bmp', 0) #read the image in grayscale
    corners = wall_corner_extraction_harris(test_img) #get corner point from the image
    
    c_dict = dict()
    idx = 0
    for i in corners:
        c_dict[idx] = (i[1], i[0])
        idx += 1
    c_dict_ = dict()
    for k, v in c_dict.items():
        c_dict_[v] = k
    
    # construct graph from the image
    G = nx.Graph()
    G.add_nodes_from([(node, {'coord': attr}) for (node, attr) in c_dict.items()]) #add nodes and corresponding coords from c_dict
    
    # check if coordinates are valid
    pixs = [test_img[i[0]][i[1]] for i in c_dict.values()]
    ratio = (1 - sum(pixs)/255/191)*100
    print('coordinates\' exactness ratio: {}%'.format(round(ratio, 3)))
    
    #get pixels from the neighbors
    def find_pixel_coord(img, point):
        if img[point[0]][point[1]] == 0:
            return point
        else:
            seq = []
            seq.append((point[0] + 1, point[1]))
            seq.append((point[0] + 1, point[1] + 1))
            seq.append((point[0], point[1] + 1))
            seq.append((point[0] - 1, point[1] + 1))
            seq.append((point[0] - 1, point[1]))
            seq.append((point[0] - 1, point[1] - 1))
            seq.append((point[0], point[1] - 1))
            seq.append((point[0] + 1, point[1] - 1))
            for i in seq:
                if img[i[0]][i[1]] != 255:
                    return i
            return (0, 0)
    if ratio > 0:
        coords = [find_pixel_coord(test_img, i) for i in c_dict.values()]
        res = [test_img[i[0]][i[1]] for i in coords]
        print('coordinates\' exactness ratio after finding points from neighbors: {}%'.format((1 - sum(res)/255/191)*100))   
    
    # reconstruct nodes dict according to coords
    nodes = dict()
    idx = 0
    for i in coords:
        nodes[idx] = (i[0], i[1])
        idx += 1
    nodes_ = dict()
    for k, v in nodes.items():
        nodes_[v] = k
    nodes = dict()
    for k, v in nodes_.items():
        nodes[v] = k
    
    return nodes, nodes_

def nodes_and_short_distance(coord1, coord2, src):

    sett = [] 
    
    #coord1: wall
    #coord2: window or door
    
    import math 
    def distt(target_x, target_y, wall_x, wall_y):
        return math.sqrt((target_x- wall_x)**2 + (target_y - wall_y)**2)
    
    for i in range(len(coord2)):
        d = dict() 
        for j in range(len(coord1)):
            target_x = coord2[i][0]
            target_y = coord2[i][1]
            wall_x = coord1[j][0]
            wall_y = coord1[j][1]             
            target_coord = [target_x, target_y]
            wall_coord = [wall_x, wall_y]  
            
            d[distt(target_x, target_y, wall_x, wall_y)] = (wall_x, wall_y)
            # d[dist((target_x, target_y), (wall_x, wall_y))] = (wall_x, wall_y)
        min_d = 99999
        for a in d.keys(): #shortest distance keeper 
            if a < min_d:
                min_d = a    
        if "win" in src: 
            if min_d < 150: 
                sett.append([[target_x, target_y], [d[min_d][0], d[min_d][1]], [min_d]])
        if "door" in src:
            if min_d < 18: 
                sett.append([[target_x, target_y], [d[min_d][0], d[min_d][1]], [min_d]])
    return sett

def binarization(filepath, fpnumber, fpnumberr, outputpath, src): 
    
    img = cv2.imread(outputpath + fpnumber + "/" + fpnumber + src + '_thin.bmp')
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)  # 이진화 시행
    cons = ~binary   
    cv2.imwrite(outputpath + fpnumber + "/" + fpnumber + src + '_bin.bmp', cons)
    

def corner_translation(filepath, outputpath, pairs, fpnumber, fpnumberr, separated_path):
    
    # img = cv2.imread(separated_path + '/' + fpnumber + '_merged.png', 0)
    new_pair = [] #인접 벽의 좌표로 문의 코너 좌표를 변환
    
    for i in range(len(pairs)):
        new_sub_x = pairs[i][1][0] #change door x-coord to nearest wall x-coord         
        new_sub_y = pairs[i][1][1] #change door y-coord to nearest wall y-coord  
        ori_sub_x = pairs[i][0][0] #keep the original door x-coord
        ori_sub_y = pairs[i][0][1] #keep the original door y-coord
        new_pair.append([[ori_sub_x, ori_sub_y], [new_sub_x, new_sub_y], [pairs[i][1][0], pairs[i][1][1]], [pairs[i][2][0]]])
  
    
    return new_pair


def raster_separation(filepath, fpnumber, fpnumberr, separated_path): 
    
    fpnumberpng = cv2.imread(filepath + '/' + fpnumber + '.png')
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
    
    canvas1[np.where((fpnumberpng == [131,232,255]).all(axis = 2))] = [0,255,255] #wall
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
    
    wall_png = cv2.imwrite(separated_path + fpnumber + '_wall.png', canvas1)
    window_png = cv2.imwrite(separated_path + fpnumber + '_window.png', canvas2)
    door_png = cv2.imwrite(separated_path + fpnumber + '_door.png', canvas3)
    merged_png = cv2.imwrite(separated_path + fpnumber + '_merged.png', canvas4)
    lift_png = cv2.imwrite(separated_path + fpnumber + '_lift.png', canvas5)
    stair_png = cv2.imwrite(separated_path + fpnumber + '_stair.png', canvas6)
    
    os.chdir(filepath)


def png2bmp(separated_path, fpnumber, fpnumberr, outputpath, src):

    img = cv2.imread(separated_path + fpnumber + src + '.png', 0) #blue laye
    img[img != 255] = 0
    cv2.imwrite(outputpath + fpnumber + '/'  + fpnumber + src + '.bmp' , img)


def thinning(filepath , fpnumber, fpnumberr, outputpath, separated_path):
    
    os.chdir(separated_path)
    
    img_name = fpnumber + '_merged.png'
    img = cv2.imread(img_name)
    
    # resize 
    # height = int(img.shape[0] * 0.8)
    # width = int(img.shape[1] * 0.8)  
    # resize_img = cv2.resize(img, (height, width), interpolation = cv2.INTER_AREA)
    # img = resize_img

    img[np.where((img == [131,232,255]).all(axis = 2))] = [0,0,0] #wall
    img[np.where((img == [0,255,255]).all(axis = 2))] = [0,0,0] #wall
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, image = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 
    BW_Original = image/255

    def pre_thinning1(image):
        rows, columns = image.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P1,P2,P3,P4,P5,P6,P7,P8 = n = neighbours(x, y, image)
                B_odd = P1 + P3 + P5 + P7
                if B_odd < 2:
                    image[y][x] = 0
                elif B_odd > 2:
                    image[y][x] = 1
                else:
                    image[y][x] = image[y][x]
        return image

    def neighbours(x,y,image):
        """Return 8-neighbours of image point P(x,y), in a clockwise order
        P4 P3 P2
        P5  P  P1
        P6 P7 P8
        """
        img = image
        x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
        plist = [ img[y][x1], img[y_1][x1], img[y_1][x], img[y_1][x_1], img[y][x_1], img[y1][x_1], img[y1][x], img[y1][x1] ] #P1 ~ P8
        return plist

    def transitions(neighbours):
        "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
        n = neighbours + neighbours[0:1]      # P1, P2, ... , P7, P8, P1
        return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P1,P2), (P2,P3), ... , (P7,P8), (P8,P1)

    def modified(image):
        "modified Thinning Algorithm"
        Image_Thinned = image.copy()  # deepcopy to protect the original image
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        changing1 = changing2 = 1        #  the points to be removed (set as 0)
        count = 0
        one_list = []
        tmp_list = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                if Image_Thinned[y][x] == 1:
                    one_list.append((x, y))

        while changing1 or changing2:   #  iterates until no further changes occur in the image
            # Step 1
            changing1 = []

            for x, y in one_list:
                P1, P2, P3, P4, P5, P6, P7, P8 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[y][x] == 1 and    # Condition 0: Point P in the object regions
                    2 <= sum(n) <= 6 and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and # Condition 2:
                    P1 * P3 * P7 == 0 and   # Condition 3
                    P1 * P5 * P7 == 0):     # Condition 4
                    changing1.append((x, y))
                else:
                    tmp_list.append((x, y))

            for x, y in changing1:
                Image_Thinned[y][x] = 0

            one_list = tmp_list
            tmp_list = []

            # Step 2
            changing2 = []

            for x, y in one_list:
                P1, P2, P3, P4, P5, P6, P7, P8 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[y][x] == 1     and    # Condition 0: Point P in the object regions
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2:
                    P1 * P5 * P7 == 0  and    # Condition 3
                    P3 * P5 * P7 == 0):         # Condition 4
                    changing2.append((x, y))
                else:
                    tmp_list.append((x, y))

            one_list = tmp_list
            tmp_list = []

            for x, y in changing2:
                Image_Thinned[y][x] = 0

            count = count + 1
            if count > 20:
                break
        return Image_Thinned
    "Apply the algorithm on images"
    BW_Skeleton = 255 - modified(pre_thinning1(BW_Original))*255     
    cv2.imwrite(outputpath + fpnumber + '/' + fpnumber + '_merged_thin.bmp' , BW_Skeleton)
    
    return BW_Skeleton


def overlay(filepath, fpnumber, fpnumberr, outputpath, separated_path): 
        
    overlay = cv2.imread(outputpath + fpnumber + '/' + fpnumber + '_merged_thin.bmp')        
    wall_background = cv2.imread(separated_path + fpnumber + "_wall.png")
    window_background = cv2.imread(separated_path + fpnumber + "_window.png")
    
    aa = overlay + wall_background
    bb = overlay + window_background 
    
    canvas_wall = np.zeros(shape=overlay.shape, dtype=np.uint8)
    canvas_wall.fill(255)
    canvas_wall[np.where((aa == [131,232,255]).all(axis = 2))] = [0,0,0] #wall
    canvas_wall[np.where((aa == [0,255,255]).all(axis = 2))] = [0,0,0] #wall
    cv2.imwrite(outputpath + fpnumber + '/' + fpnumber + '_wall_thin.bmp', canvas_wall)
    
    
    canvas_window = np.zeros(shape=overlay.shape, dtype=np.uint8)
    canvas_window.fill(255)
    canvas_window[np.where((bb == [0,0,255]).all(axis = 2))] = [0,0,0] #
    cv2.imwrite(outputpath + fpnumber + '/' + fpnumber + '_window_thin.bmp', canvas_window)


def line_generation(outputpath, fpnumber, fpnumberr, src):
    
    def r2v(outputpath, file_name, fpnumber, fpnumberr, src):   
        outresult = file_name.split(".")[0]+".geojson"
        
        if 'win' in str(file_name):
            os.chdir(outputpath + fpnumber)
            outresult = str(fpnumber) + '_window.geojson'
        elif 'merged' in str(file_name):
            os.chdir(outputpath + fpnumber)
            outresult = str(fpnumber) + '_merged.geojson'
        elif 'stair' in src:
            os.chdir(outputpath + fpnumber)
            outresult = str(fpnumber) + '_stair.geojson'
        elif 'wall' in str(file_name):
            os.chdir(outputpath + fpnumber)
            outresult = str(fpnumber) + '_wall.geojson'
        else:
            os.chdir(outputpath + fpnumber)
            outresult = str(fpnumber) + '_lift.geojson'
        
        src_ds = gdal.Open(outputpath + fpnumber + '/' + fpnumber + src) #read image
        originX, pixelWidth, x_rotation, originY, y_rotation, pixelHeight = src_ds.GetGeoTransform() #Image GeoTransform
        
        # Image EPSG
        proj = osr.SpatialReference(wkt=src_ds.GetProjection())
        crs = "EPSG:{}".format(proj.GetAttrValue('AUTHORITY',1))
        
        img = src_ds.ReadAsArray() #Image as array
        imgbin = img == 255 #Create a boolean binary image
        imgbin = imgbin[1, :, :]
        img_thin = thin(imgbin) #The thinned image
        if img_thin.sum() == 0:
            return False
        img_thin = img_as_ubyte(img_thin) #Convert to 8-bit uint (0-255 range)
        img_thin = np.where(img_thin == 255) #this will return the indices of white pixels
        
        
        # Calculate the center point of each cell
        points = []
        for i,idY in enumerate(img_thin[0]):
            idX = img_thin[1][i]
            cX = idX
            cY = idY
            points.append((int(cX), int(cY)))
       
        maxD = np.sqrt(2.0*pixelWidth*pixelWidth)
        tree = spatial.cKDTree(points)
        groups = tree.query_ball_tree(tree, maxD)
        
        out_lines = []
        for x, ptlist in enumerate(groups):
            for pt in ptlist:
                if pt>x:            
                    out_lines.append(string([point(points[x]), point(points[pt])]))    
        
        # Merge the contiguous lines 
        merged_lines = linemerge(MultiLineString(out_lines))
        
        # Define the crs
        crs = {
            "type": "name",
            "properties": {
                "name": crs
            }
        }
        
        # Create a feature collection of linestrings
        feature_collection = []
        if isinstance(merged_lines,string):
            merged_lines = merged_lines.simplify(maxD, True)
            feature_collection.append(geojson.Feature(geometry=geojson.LineString(merged_lines.coords[:])))
        else:    
            for line in merged_lines:
                # Simplify the result to try to remove zigzag effect
                line = line.simplify(maxD, True)
                feature_collection.append(geojson.Feature(geometry=geojson.LineString(line.coords[:])))                                  
                       
        feature_collection = geojson.FeatureCollection(feature_collection,crs=crs)
        
        # Save the output to a geosjon file
        with open(outresult, 'w') as outfile:
            os.chdir(outputpath + fpnumber + '/')
            geojson.dump(feature_collection, outfile)
        return feature_collection
               
    if src == None :      
        #merged line generation
        merged_file_name = str(fpnumber) + '/' + str(fpnumber) + '_merged_bin.bmp'
        src_merge = '_merged_bin.bmp'
        merge_vec = r2v(outputpath, merged_file_name, fpnumber, fpnumberr, src_merge)
        merge_vec = [shape(i['geometry']) for i in merge_vec['features']]
        
        #window line generation
        win_file_name = str(fpnumber) + '/' + str(fpnumber) + '_window_bin.bmp'
        src_window = '_window_bin.bmp'
        win_vec = r2v(outputpath, win_file_name, fpnumber, fpnumberr, src_window)
        
        #wall (only) line generation 
        wall_file_name = str(fpnumber) + '/' + str(fpnumber) + '_wall_bin.bmp'
        src_wall = '_wall_bin.bmp'
        wall_vec = r2v(outputpath, wall_file_name, fpnumber, fpnumberr, src_wall)
        wall_vec = [shape(i['geometry']) for i in wall_vec['features']]
        
        if win_vec:
            win_vec = [shape(i['geometry']) for i in win_vec['features']]
            with open(outputpath + fpnumber + "/" + fpnumber + "_wall.geojson") as f:
                linevec = geojson.load(f)
            linevec = linevec['features']
            linevec = [shape(linevec[i]['geometry']) for i in range(len(linevec))]  #convert geojson obj to shapely obj   
        else:
           win_vec = False 
           return
        os.chdir(outputpath)
    
    if src != None: 
        #stair & lift   
        file_name = str(fpnumber) + src + '_bin.bmp'
        srccc = src + '_bin.bmp'
        
        line_vec = r2v(outputpath, file_name, fpnumber, fpnumberr, srccc)
        line_vec = [shape(i['geometry']) for i in line_vec['features']]

def new_wall(outputpath, fpnumber, fpnumberr):
    
    wall_line = gpd.read_file(outputpath + fpnumber + '/' + fpnumber + '_wall.geojson').geometry
    wall_point = [list(wall_line[i].boundary) for i in range(len(wall_line))]
    wall_points = []
    for i in range(len(wall_point)):
        for j in range(len(wall_point[i])):
            x_coord = wall_point[i][j].xy[0][0]
            y_coord = wall_point[i][j].xy[1][0]
            wall_points.append([x_coord, y_coord])
            
    unique = [i for i in wall_points if wall_points.count(i) == 1]
    unique_wall_sp = [point(i) for i in unique]
    connect2 = []
    for i in unique_wall_sp: 
        idx = unique_wall_sp.index(i)
        for j in unique_wall_sp[idx:]:
            if j not in [connect2[i][1] for i in range(len(connect2))]:
                dist = i.distance(j)
                min_d = 9999
                if dist > 1 and dist < min_d: 
                    min_d = dist     
                if min_d < 13:
                    connect2.append([i, j, dist])   
                    
                    
    new_wall = []
    for i in range(len(connect2)):
        ori_crd = connect2[i][0]
        connect_crd = connect2[i][1]
        new_wall_line = string([ori_crd, connect_crd])
        new_wall.append(new_wall_line)
    
    
    new_wall_gj = [] # door_line을 geojson으로 저장
    for i in range(len(new_wall)):
        new_wall_gj.append(Feature(geometry=new_wall[i], properties={}))
    subject = FeatureCollection(new_wall_gj)
    with open(outputpath + fpnumber + '/' + fpnumber + '_walll_new_line.geojson', 'w') as f:
        dump(subject, f)
    
    
    return new_wall_gj
        

def square_check(filepath, fpnumber, fpnumberr):
    
    min_size = 3309
    img = Image.open(filepath + fpnumberr)
    img_size = img.size
    x = img_size[0]
    y = img_size[1]
    
    if x != y: 
        size = max(min_size, x, y)    
        resized_img = Image.new(mode = 'RGB', size = (size, size), color = (0, 0, 255))
        offset = (round((abs(x - size)) / 2), round((abs(y - size)) / 2))
        resized_img.paste(img, offset)
        resized_img.save(filepath + fpnumberr)
    else:
        None


def sp_corner(outputpath, fpnumber, fpnumberr, src):
    
    filename = outputpath + fpnumber + '/' + fpnumber + '_door.bmp' 
    img = cv2.imread(filename, 0)
    height, width = img.shape
    
    zero = empty_check(fpnumber, fpnumberr, outputpath, src='_window')
    if zero == 0: 
        return 
    
    with open(outputpath + fpnumber + '/' + fpnumber + src + '.geojson') as f:
        vec = gj.load(f)
    vec_line = [vec['features'][i]['geometry'] for i in range(len(vec['features']))]#geojson obj
    vec_line = [shape(vec_line[i]) for i in range(len(vec_line))] #shapely obj
    
    affine_m = [1, 0, 0, -1, 0, width]
    sp_coord = []
    
    for i in range(len(vec_line)):
        # if i == 202:
        #     print('here')
        final_line = af(vec_line[i], affine_m)
        if len(final_line.boundary) != 0:
            start, ends = final_line.boundary #각 라인의 양끝점 좌표를 shapely.geometry.point.Point 형태로 받음 
            start_x = int(start.xy[0][0])
            start_y = int(start.xy[1][0])
            ends_x = int(ends.xy[0][0])
            ends_y = int(ends.xy[1][0])
            
            sp_coord.append([start_x, start_y])
            sp_coord.append([ends_x, ends_y])
            
    
    if src == '_wall':  #각 라인의 양끝점 뿐만 아니라 분기점의 좌표도 넣기 
        for i in range(len(vec_line)):
            final_line = af(vec_line[i], affine_m)
            cc = list(final_line.coords)
            for j in range(int(len(cc)/2)):
                coords = cc[j]
                x_crd = coords[0]
                y_crd = coords[1]
                sp_coord.append([x_crd, y_crd])
    
    return sp_coord, vec_line


def contour_polygon(separated_path, fpnumber, fpnumberr, outputpath, src): 

    filename = separated_path + fpnumber
    img = cv2.imread(filename + src + '.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    test = binary
    
    contours, h = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if 'window' in src: 
        image = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    if 'door' in src:
        image = cv2.drawContours(img, contours, -1, (0,255,0), 1)
            
    if 'door' in src: 
        canvas = np.zeros(shape=img.shape, dtype=np.uint8)
        canvas.fill(255)
        canvas[np.where((image == [0,0,0]).all(axis = 2))] = [255,255,255]
        canvas[np.where((image == [0,255,0]).all(axis = 2))] = [0,255,0] #door+window 
        cv2.imwrite(outputpath + fpnumber + '/' + fpnumber + '_door_contour.png', canvas)
    
    if 'window' in src: 
        canvas = np.zeros(shape=img.shape, dtype=np.uint8)
        canvas.fill(255)
        canvas[np.where((image == [0,0,0]).all(axis = 2))] = [255,255,255]
        canvas[np.where((image == [0,0,255]).all(axis = 2))] = [0,0,0] #window
        canvas[np.where((image == [0,255,0]).all(axis = 2))] = [0,255,0] #contour
        cv2.imwrite(outputpath + fpnumber + '/' + fpnumber + '_window_contour.png', canvas)
    
    poly_objs = []
    bigger_objs = []
    for i in range(len(contours)):
        if (i > 0) and (len(contours[i])) > 2:
            poly_objs.append(Polygon(np.squeeze(contours[i])).buffer(3))
            bigger_objs.append(Polygon(np.squeeze(contours[i])).buffer(4))
    
    return poly_objs, bigger_objs



def block_and_connect(separated_path, outputpath, fpnumber, fpnumberr, coord1, coord2, polygon, pair, src):
    
    #coord1: wall
    #coord2: window or door 
    
    filename = separated_path + fpnumber
    img = cv2.imread(filename + '_merged.png')
    

    
    affine_m = [1, 0, 0, -1, 0, img.shape[0]]
    sub_point = [] #change subject coords(window or door) to shapley Points 
    for i in range(len(coord2)): 
        sub_x = coord2[i][0]
        sub_y = coord2[i][1]
        a = point(sub_x, sub_y)
        apoint = af(a, affine_m)
        sub_point.append(apoint) # calibrate coordinates using affine matrix 
        # sub_point = shapely points of subject(window or door) corners 
    

    
    wall_point = [] 
    for i in range(len(coord1)): 
        wall_x = coord1[i][0]
        wall_y = coord1[i][1]
        apoint = af(point(wall_x, wall_y), affine_m)
        wall_point.append(apoint)
    
    sub_dict = dict()
    for i in range(len(polygon)):
        sub_list = [] 
        for j in range(len(sub_point)): 
            if polygon[i].contains(sub_point[j]) == True: 
                sub_list.append(j)
        sub_dict[i] = sub_list
        # sub_dict = subject_corners in the same subject polygon
    

    
    def vis(sub_polygons, sub_points, wall_points, src):
        polygon = gpd.GeoDataFrame(sub_polygons, columns = ['geometry'])
        sub_points = gpd.GeoDataFrame(sub_points, columns = ['geometry'])
        wall_points = gpd.GeoDataFrame(wall_points, columns = ['geometry'])
        polygon.to_file(outputpath + fpnumber + '/' + fpnumber + src + '_polys.shp')
        sub_points.to_file(outputpath + fpnumber + '/' + fpnumber + src + '_corner.shp')
        wall_points.to_file(outputpath + fpnumber + '/' + fpnumber + '_wall_corner.shp')
    
    vis(polygon, sub_point, wall_point, src)
 
    close = dict() # 폐합하는 부분 
    for i in range(len(sub_dict)):
        av = [] 
        for j in range(len((sub_dict[i]))): 
            idx = sub_dict[i][j] #door 코너의 인덱스
            crd = coord2[idx] #해당 인덱스의 좌표값    
            for k in range(len(pair)): #new_set = [문원래좌표, 바뀐거, 인접벽, 거리]
                if crd == pair[k][0]: #문원래 좌표랑 맞는게 있으면 
                    crd_n = pair[k][1] #바뀐 좌표로 바꿔라
                    av.append(crd_n)
        close[i] = av
    

    
    # 인접 벽의 좌표들로 바뀐 문 좌표들을 문 인덱스별로 묶음

    sub_line = [] 
    if "door" in src:
        for i in range(len(close)):
            sng = [] 
            for j in range(len(close[i])):
                ax = close[i][j][0]
                ay = close[i][j][1]
                a = point(int(ax), int(ay))
                apoint = af(a, affine_m) 
                sng.append(apoint)
            if len(sng) > 1: 
                line = string(sng)
                sub_line.append(line)

    if "win" in src: 
        for i in range(len(close)):
            sng = [] 
            for j in range(len(close[i])):
                ax = close[i][j][0]
                ay = close[i][j][1]
                a = point(int(ax), int(ay))
                apoint = af(a, affine_m) 
                sng.append(apoint)
            if len(sng) > 1:   #and len(sng) < 15: 
                line = string(sng)
                sub_line.append(line)
    
    sub_gj = [] # door_line을 geojson으로 저장
    for i in range(len(sub_line)):
        sub_gj.append(Feature(geometry=sub_line[i], properties={}))
    subject = FeatureCollection(sub_gj)
    with open(outputpath + fpnumber + '/' + fpnumber + src + '_line.geojson', 'w') as f:
        dump(subject, f)
    
    
    
    with open(outputpath + fpnumber + '/' + fpnumber + '_wall.geojson') as f:
        wall_v = gj.load(f)
    wall_vecs = [wall_v['features'][i]['geometry'] for i in range(len(wall_v['features']))] #geojson obj
    wall_vecs = [shape(wall_vecs[i]) for i in range(len(wall_vecs))] #shapely obj
    wall_lines = sp.ops.linemerge(wall_vecs)

    
    return sub_gj

def polygonization(outputpath, fpnumber, fpnumberr):
   
    def split_wall(wall_line, wall_point, wall_splited):

        # shapely library의 union(전체 객체를 multi part 객체로 union)과 split 함수 사용
        line = wall_line.geometry.unary_union
        point = wall_point.geometry.unary_union
        split_result = split(line, point)
        
        # fiona library 활용하여 shapefile write
        schema = {'geometry': 'LineString','properties': {'id': 'int'}}
        with fiona.open(wall_splited, 'w', 'ESRI Shapefile', schema) as c:
            for poly_id,polyline in enumerate(split_result):
                c.write({
                    'geometry': mapping(polyline),
                    'properties': {'id': poly_id},
                })
        return split_result 
    
    def read_gj(line):
        with open(line) as f:
            gj_file = gj.load(f) 
        return gj_file
    
    def merge2multilines(all_features): 
        src_vecs = [all_features[i]['geometry'] for i in range(len(all_features))]#geojson obj
        src_vecs = [shape(src_vecs[i]) for i in range(len(src_vecs))]#shapely obj
        src_multilines = linemerge(src_vecs)
        return src_multilines
    
    def save_as_shp(geometry_list, outputpath, fpnumber, src):
        geometrys = gpd.GeoDataFrame(geometry_list, columns = ['geometry'])
        if not os.path.exists(outputpath + fpnumber + '/polygonize'):
            os.mkdir(outputpath + fpnumber + '/polygonize')
        geometrys.to_file(outputpath + fpnumber + '/polygonize/' + fpnumber + src + '.shp')
    
    
    
    files = os.listdir(outputpath + fpnumber)    
    for file in files:
        if '_wall.geojson' in str(file): 
            wall_line = str(outputpath) + str(fpnumber) + '/' + str(fpnumber) + '_wall.geojson'
            wall_f = read_gj(wall_line)
            wall_features = wall_f['features']
        
        if '_window_line.geojson' in str(file):
            window_line = str(outputpath) + str(fpnumber) + '/' + str(fpnumber) + '_window_line.geojson'
            window_f = read_gj(window_line)
            window_features = window_f['features']
           
        if '_door_line.geojson' in str(file):
            door_line = str(outputpath) + str(fpnumber) + '/' + str(fpnumber) + '_door_line.geojson'
            door_f = read_gj(door_line)
            door_features = door_f['features']
        
        if '_stair.geojson' in str(file): 
            stair_line = str(outputpath) + str(fpnumber) + '/' + str(fpnumber) + '_stair.geojson'
            stair_f = read_gj(stair_line)
            stair_features = stair_f['features']
            
        if '_lift.geojson' in str(file):        
            lift_line = str(outputpath) + str(fpnumber) + '/' + str(fpnumber) + '_lift.geojson'
            lift_f = read_gj(lift_line)
            lift_features = lift_f['features']
        
        if '_walll_new_line' in str(file):        
            wall_new_line = str(outputpath) + str(fpnumber) + '/' + str(fpnumber) + '_walll_new_line.geojson'
            wall_new_f = read_gj(wall_new_line)
            wall_new_features = wall_new_f['features']
      
            
    merged_features = [] #wall, door, window, new_wall
    merged_lift = []  #lift
    merged_stair = [] #stair 
    
    path = str(outputpath) + str(fpnumber) + '/'
    ffiles = os.listdir(path)
    
    
    wwall_file_name = str(fpnumber) + '_wall.geojson'
    wwall_new_file_name = str(fpnumber) + '_walll_new_line.geojson'
    wwindow_file_name = str(fpnumber) + '_window_line.geojson'
    ddoor_file_name = str(fpnumber) + '_door_line.geojson'
    sstair_file_name = str(fpnumber) + '_stair.geojson'
    llift_file_name = str(fpnumber) + '_lift.geojson'
    
    if wwall_file_name in ffiles:
        for i in range(len(wall_features)):
            if wall_features[i]['geometry']['type'] == 'LineString':
                merged_features.append(wall_features[i])
                
    if wwindow_file_name in ffiles:            
        for i in range(len(window_features)):
            if window_features[i]['geometry']['type'] == 'LineString':
                merged_features.append(window_features[i])
    
    if ddoor_file_name in ffiles:        
        for i in range(len(door_features)):
            if door_features[i]['geometry']['type'] == 'LineString':
                merged_features.append(door_features[i])
    
    if wwall_new_file_name in ffiles: 
        for i in range(len(wall_new_features)):
            if wall_new_features[i]['geometry']['type'] == 'LineString':
                merged_features.append(wall_new_features[i])
    
    if sstair_file_name in ffiles: 
        for i in range(len(stair_features)):
            if stair_features[i]['geometry']['type'] == 'LineString':
                merged_stair.append(stair_features[i])
                
        stairs = merge2multilines(merged_stair)
        stair_list = [i for i in polygonize(stairs)]
        
        new_stair_list = []
        for stair in stair_list: 
            area = stair.area 
            if area > 600 : 
                new_stair_list.append(stair)
        
        if len(new_stair_list) > 0: 
            save_as_shp(new_stair_list, outputpath, fpnumber, src='_stair_polygons')
    
    if llift_file_name in ffiles: 
        for i in range(len(lift_features)):
            if lift_features[i]['geometry']['type'] == 'LineString':
                merged_lift.append(lift_features[i])
                
        lifts = merge2multilines(merged_lift)
        lift_list = [i for i in polygonize(lifts)]
        
        new_lift_list = []
        for lift in lift_list:
            area = lift.area
            if area > 600 : 
                new_lift_list.append(lift)
         
        if len(new_lift_list) > 0:
            save_as_shp(new_lift_list, outputpath, fpnumber, src='_lift_polygons')
        
            
    
    all_multi = merge2multilines(merged_features) #merge wall/window/door LineStrings into one MultiLineString 
    lines = gpd.GeoDataFrame(all_multi, columns = ['geometry'])
    
    os.chdir(outputpath)
    wall_corner = gpd.read_file(fpnumber + '/' + fpnumber + '_wall_corner.shp')
    new_polys = split_wall(lines, wall_corner, outputpath + fpnumber + '/' + fpnumber + '_wall_splited.shp')
    new_polys_list = [i for i in polygonize(new_polys)]
    
    
    save_as_shp(merged_features, outputpath, fpnumber, src='_merged_line')
    save_as_shp(new_polys_list, outputpath, fpnumber, src='_new_polys')
        

def bi_bbox(filepath, outputpath, separated_path, fpnumber, fpnumberr, src): 

    bi_img = cv2.imread(separated_path + '/' + fpnumber + src + '.png', 0)
    ori_bi = bi_img.copy()
    bi = ~bi_img
    
    ret, bi_bin = cv2.threshold(bi, 30, 255, 0) #bi binary
    bi_contours, hierarchy = cv2.findContours(bi_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(bi_contours) < 1: #if there's no bi, just get out
        return
    
    bi_canvas1 = np.zeros(shape=bi_img.shape, dtype=np.uint8)
    bi_canvas1.fill(255)
    bi_canvas2 = np.zeros(shape=bi_img.shape, dtype=np.uint8)
    bi_canvas2.fill(255)
    
    # contourss = []
    # for cnt in bi_contours: 
    #     area = cv2.contourArea(cnt)
    #     if area > 300: 
    #         contourss.append(cnt)
    
    "For Labeling Data"
    if "t" not in fpnumber: 
        cnt_pt = []
        for cnt in bi_contours:
            if cnt.size > 40 : #200 - 400 사이에서 조절 for real DL output
                gig_img = cv2.drawContours(bi_img, [cnt], 0, (0, 0, 0), 3)  # blue   
                cnt_pt.append(cnt)
            else: #####조건문에 세부적으로 추가 필요 
                continue 
        bi_cont = cv2.imwrite(outputpath + fpnumber + '/' + fpnumber + src + '_cont.png', gig_img)
        
        #bounding box
        for i in range(len(cnt_pt)):
            dks = cnt_pt[i]
            rect = cv2.minAreaRect(dks)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            bnd_img1 = cv2.drawContours(ori_bi,[box],0,(0,0,0),5)
            bnd_img2 = cv2.drawContours(bi_canvas2,[box],0,(0,0,0),5)
            
        bi_bnd1 = cv2.imwrite(outputpath + fpnumber + '/' +  fpnumber + src + '_bnd1.png', bnd_img1)
        bi_bnd2 = cv2.imwrite(outputpath + fpnumber + '/' + fpnumber + src + '_bnd2.png', bnd_img2)

    return bi_bnd2

def moveup(final_path, outputpath, fpnumber):

    files = os.listdir(outputpath + fpnumber + '/')
    wall_file_name = str(fpnumber) + '_wall.dbf'
    wall_new_file_name = str(fpnumber) + '_new_wall.dbf'
    window_file_name = str(fpnumber) + '_window_line.dbf'
    door_file_name = str(fpnumber) + '_door_line.dbf'
    stair_file_name = str(fpnumber) + '_stair.dbf'
    lift_file_name = str(fpnumber) + '_lift.dbf'
    
    src = str(outputpath) + str(fpnumber) + '/'
    dst = final_path 
    
    if wall_file_name in files:
        shutil.move(src + wall_file_name, dst + wall_file_name)
    if wall_new_file_name in files:
        shutil.move(src + wall_new_file_name, dst + wall_new_file_name)
    if window_file_name in files:
        shutil.move(src + window_file_name, dst + window_file_name)
    if door_file_name in files:
        shutil.move(src + door_file_name, dst + door_file_name)
    if stair_file_name in files:
        shutil.move(src + stair_file_name, dst + stair_file_name)
    if lift_file_name in files:
        shutil.move(src + lift_file_name, dst + lift_file_name)


def cleanup(clean_path, outputpath, fpnumber):
    
    os.chdir(outputpath + fpnumber)
    files = os.listdir()
    for file in files: 
        if fpnumber in file: 
            shutil.move(file, clean_path)
    

def polygon2coord(outputpath, fpnumber, polygons):
    
    filename = outputpath + fpnumber + '/' + fpnumber + '_door.bmp' 
    img = cv2.imread(filename, 0)
    height, width = img.shape
    affine_m = [1, 0, 0, -1, 0, width]
    
    polygons = gpd.GeoDataFrame(polygons, columns = ['geometry'])
    polygons.to_file(outputpath + fpnumber + '/' + fpnumber + '_door_polys.shp')
    
    door_poly = gpd.read_file(outputpath + fpnumber + '/' + fpnumber + '_door_polys.shp').geometry
    
    
    door_polys = []
    for door in door_poly: #change gpd points to shapely points 
        door = af(door, affine_m)
        a_door = Polygon(door) 
        door_polys.append(a_door)
    
    
    door_harris = []
    for i in range(len(door_polys)):
        coords = list(door_polys[i].exterior.coords)
        for j in range(len(coords)):
            x_coord, y_coord = coords[j]
            door_harris.append([int(x_coord), int(y_coord)])
        
    return door_harris


def add_field (outputpath, fpnumber): 
       
    
    os.chdir(outputpath + fpnumber + '/polygonize/')
    polygon_files = os.listdir(outputpath + fpnumber + '/polygonize/')
    
    for file in polygon_files: 
        if 'polys' in file: 
            polys_gdf = gpd.read_file(fpnumber + '_new_polys.shp')
            polys_gdf['space_id'] = polys_gdf.FID
            polys_gdf['auto_YN'] = 0
            polys_gdf['floor_num'] = None
            polys_gdf['height'] = None
            polys_gdf['usage'] = None
            polys_gdf['name'] = None
            polys_gdf['hollow_YN'] = None
            polys_gdf['stair_id'] = None
            polys_gdf['lift_id'] = None
            schema = {'geometry': 'Polygon',
                      'properties': {'FID': 'int:9', 'space_id': 'int:9', 'auto_YN': 'int:1', 
                                     'floor_num': 'int:2', 'height': 'float', 'usage': 'int:2', 
                                     'name': 'str', 'hollow_YN': 'int:1',
                                     'stair_id': 'int:9', 'lift_id': 'int:9'}}
            polys_gdf.to_file(fpnumber + '_new_polys.shp', driver='Shapefile', schema=schema)
        
        if 'stair' in file: 
    
            stair_gdf = gpd.read_file(fpnumber + '_stair_polygons.shp')
            stair_gdf['stair_id'] = stair_gdf.FID
            stair_gdf['width'] = None 
            stair_gdf['depth'] = None 
            stair_gdf['auto_YN'] = 0
            stair_gdf['floor_num'] = None
            stair_gdf['height'] = None
            stair_gdf['upper_id'] = None
            stair_gdf['lower_id'] = None
            stair_gdf['template'] = None
            stair_gdf['steps'] = None
            schema = {'geometry': 'Polygon',
                      'properties': {'FID': 'int:9', 'stair_id': 'int:9', 'width': 'float',
                                     'depth': 'float','auto_YN': 'int:1', 'floor_num': 'int:2', 
                                     'height': 'float', 'upper_id': 'int:10', 
                                     'lower_id': 'int:10', 'template': 'int:2',
                                     'steps': 'int:2'}}
            stair_gdf.to_file(fpnumber + '_stair_polygons.shp', driver='Shapefile', schema=schema)
    
        if 'lift' in file:
            lift_gdf = gpd.read_file(fpnumber + '_lift_polygons.shp')
            lift_gdf['lift_id'] = lift_gdf.FID
            lift_gdf['width'] = None 
            lift_gdf['depth'] = None 
            lift_gdf['auto_YN'] = 0
            lift_gdf['floor_num'] = None
            lift_gdf['height'] = None
            lift_gdf['upper_id'] = None
            lift_gdf['lower_id'] = None
            lift_gdf['template'] = None
            lift_gdf['door_id'] = None
            schema = {'geometry': 'Polygon',
                      'properties': {'FID': 'int:9', 'lift_id': 'int:9', 'width': 'float',
                                     'depth': 'float','auto_YN': 'int:1', 'floor_num': 'int:2', 
                                     'height': 'float', 'upper_id': 'int:10', 
                                     'lower_id': 'int:10', 'template': 'int:2',
                                     'door_id': 'int:10'}}
            lift_gdf.to_file(fpnumber + '_lift_polygons.shp', driver='Shapefile', schema=schema)
    
    
    os.chdir(outputpath + fpnumber + '/')
    vector_path = outputpath + fpnumber + '/vector_output/'
    line_files = os.listdir(outputpath + fpnumber + '/')
    
    for file in line_files: 
        if 'wall' in file:
            wall_gdf = gpd.read_file(fpnumber + '_wall.geojson')
            wall_gdf['wall_id'] = range(len(wall_gdf))
            wall_gdf['auto_YN'] = 0
            wall_gdf['floor_num'] = None
            wall_gdf['height'] = None
            wall_gdf['virtual_YN'] = None
            schema = {'geometry': 'LineString',
                      'properties': {'wall_id': 'int:9', 'auto_YN': 'int:1', 
                                     'floor_num': 'int:2', 'height': 'float', 'virtual_YN': 'int:1'}}
            wall_gdf.to_file(vector_path + fpnumber + '_wall.shp', driver='Shapefile', schema=schema)
            
        if 'walll' in file:
            wall_gdf = gpd.read_file(fpnumber + '_walll_new_line.geojson')
            wall_gdf['wall_id'] = range(len(wall_gdf))
            wall_gdf['auto_YN'] = 0
            wall_gdf['floor_num'] = None
            wall_gdf['height'] = None
            wall_gdf['virtual_YN'] = None
            schema = {'geometry': 'LineString',
                      'properties': {'wall_id': 'int:9', 'auto_YN': 'int:1', 
                                     'floor_num': 'int:2', 'height': 'float', 'virtual_YN': 'int:1'}}
            wall_gdf.to_file(vector_path + fpnumber + '_new_wall.shp', driver='Shapefile', schema=schema)
    
        if 'wind' in file: 
            window_gdf = gpd.read_file(fpnumber + '_window_line.geojson')
            window_gdf['window_id'] = range(len(window_gdf))
            window_gdf['auto_YN'] = 0
            window_gdf['floor_num'] = None
            window_gdf['height'] = None
            window_gdf['lower'] = None
            window_gdf['upper'] = None
            window_gdf['template'] = None
            schema = {'geometry': 'LineString',
                      'properties': {'window_id': 'int:9', 'auto_YN': 'int:1', 
                                     'floor_num': 'int:2', 'height': 'float', 'lower': 'float', 
                                     'upper': 'float', 'upper': 'float', 'template': 'int:2'}}
            window_gdf.to_file(vector_path + fpnumber + '_window_line.shp', driver='Shapefile', schema=schema)
    
        if 'door' in file: 
            door_gdf = gpd.read_file(fpnumber + '_door_line.geojson')
            door_gdf['door_id'] = range(len(door_gdf))
            door_gdf['auto_YN'] = 0
            door_gdf['floor_num'] = None
            door_gdf['height'] = None
            door_gdf['lower'] = None
            door_gdf['upper'] = None
            door_gdf['template'] = None
            door_gdf['d_direc'] = 2
            door_gdf['lift_YN'] = None
            schema = {'geometry': 'LineString',
                      'properties': {'door_id': 'int:9', 'auto_YN': 'int:1', 
                                     'floor_num': 'int:2', 'height': 'float', 'lower': 'float', 
                                     'upper': 'float', 'upper': 'float', 'template': 'int:2',
                                     'd_direc':'int:1', 'lift_YN':'int:1'}}
            door_gdf.to_file(vector_path + fpnumber + '_door_line.shp', driver='Shapefile', schema=schema)
    


# #===========================================================================================================

def main(fpnumber):                       
    
    mainpath = 'C:/Users/user/Desktop/module1_2/'   
    filepath = 'C:/Users/user/Desktop/module1_2/test_input/'
    outputpath = 'C:/Users/user/Desktop/module1_2/test_output/'
    separated_path = 'C:/Users/user/Desktop/module1_2/separated_output/'
    final_path = 'C:/Users/user/Desktop/module1_2/test_output/' + str(fpnumber) + '/vector_output/'
    clean_path = 'C:/Users/user/Desktop/module1_2/test_output/' + str(fpnumber) + '/temp/'
    
    fpnumber = fpnumber
    fpnumberr = fpnumber + '.png'   
    
    print("  ")
    print("   >> " + str(fpnumber) + "  START ")    
    print("  ")
    
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
    if not os.path.exists(outputpath + '/' + fpnumber):
        os.mkdir(outputpath + '/' + fpnumber)
    if not os.path.exists(final_path):
        os.mkdir(final_path)
    if not os.path.exists(clean_path):
        os.mkdir(clean_path)
    
    square_check(filepath, fpnumber, fpnumberr)
    print("1 / 17")
    
    raster_separation(filepath, fpnumber, fpnumberr, separated_path)
    print("2 / 17")
    
    png2bmp(separated_path, fpnumber, fpnumberr, outputpath, src='_door')
    png2bmp(separated_path, fpnumber, fpnumberr, outputpath, src='_window')
    print("3 / 17")
    
    
    thinning(filepath, fpnumber, fpnumberr, outputpath, separated_path)
    print("4 / 17")
    
    overlay(filepath, fpnumber, fpnumberr, outputpath, separated_path)
    print("5 / 17")
    
    binarization(filepath, fpnumber, fpnumberr, outputpath, src='_merged')
    binarization(filepath, fpnumber, fpnumberr, outputpath, src='_wall')
    binarization(filepath, fpnumber, fpnumberr, outputpath, src='_window')
    print("6 / 17")
  
    os.chdir(mainpath)
    line_generation(outputpath, fpnumber, fpnumberr, src=None)
    print("7 / 17")
    
    
    door_polygon, bigger_door = contour_polygon(separated_path, fpnumber, fpnumberr, outputpath, src='_door')
    window_polygon, bigger_window = contour_polygon(separated_path, fpnumber, fpnumberr, outputpath, src='_window')
    print("8 / 17")
    
    door_harris_coord = polygon2coord(outputpath, fpnumber, door_polygon)
    print("9 / 17")
    
    
    # door_harris_coord = harris_tolist(door_harris)
    # print("9 / 17")
    
    wall_coord, wall_sp_line = sp_corner(outputpath, fpnumber, fpnumberr, src='_wall')
    window_coord, window_sp_line = sp_corner(outputpath, fpnumber, fpnumberr, src='_window')
    print("10 / 17")
    
    wall_n_door_pairs = nodes_and_short_distance(wall_coord, door_harris_coord, src="door") 
    wall_n_window_pairs = nodes_and_short_distance(wall_coord, window_coord, src="window")
    print("11 / 17")
    
    wall_n_door_new_pairs = corner_translation(filepath, outputpath, wall_n_door_pairs, fpnumber, fpnumberr, separated_path)
    wall_n_window_new_pairs = corner_translation(filepath, outputpath, wall_n_window_pairs, fpnumber, fpnumberr, separated_path)
    print("12 / 17")

    
    door_gj = block_and_connect(separated_path, outputpath, fpnumber, fpnumberr, wall_coord, door_harris_coord, bigger_door, wall_n_door_new_pairs, src='_door')
    window_gj = block_and_connect(separated_path, outputpath, fpnumber, fpnumberr, wall_coord, window_coord, window_polygon, wall_n_window_new_pairs, src='_window')
    new_wall_gj = new_wall(outputpath, fpnumber, fpnumberr)
    print("13 / 17")

    
    stair_check = bi_check(separated_path, fpnumber, fpnumberr, src='_stair')
    lift_check = bi_check(separated_path, fpnumber, fpnumberr, src='_lift') 
    
    if stair_check > 0.01: 
        stair_bb = bi_bbox(filepath, outputpath, separated_path, fpnumber, fpnumberr, src='_stair')
        bi_thinning(fpnumber, fpnumberr, outputpath, separated_path, src='_stair')
        binarization(filepath, fpnumber, fpnumberr, outputpath, src='_stair')
        line_generation(outputpath, fpnumber, fpnumberr, src='_stair')
        print("14 / 17")
    else: 
        print("* No stair")
    
        
    if lift_check > 0.01: 
        lift_bb = bi_bbox(filepath, outputpath, separated_path, fpnumber, fpnumberr, src='_lift')
        bi_thinning(fpnumber, fpnumberr, outputpath, separated_path, src='_lift')
        binarization(filepath, fpnumber, fpnumberr, outputpath, src='_lift')
        line_generation(outputpath, fpnumber, fpnumberr, src='_lift')
        print("15 / 17")
    else: 
        print("* No lift")
    
    
    polygonization(outputpath, fpnumber, fpnumberr)
    print("16 / 17")
    
    add_field(outputpath, fpnumber)
    # moveup(final_path, outputpath, fpnumber)
    cleanup(clean_path, outputpath, fpnumber)
    print("17 / 17")
    print(str(fpnumber) + " DONE")
    

    
#=========================================================================================================================

filepath = 'C:/Users/user/Desktop/module1_2/test_intput/'
outpath = 'C:/Users/user/Desktop/module1_2/test_output/'

fplist = os.listdir(filepath)
donelist = os.listdir(outpath)

fplist_ = []
for fp in fplist:
    fpp = fp[:-4]
    fplist_.append(fpp)  
    realist = list(set(fplist_) - set(donelist))

if __name__=='__main__':   
    for fp in realist: 
        print(" ")
        print("======== {} out of {} DONE ======== ".format(realist.index(fp), len(realist)))
        start_time = time.time()
        main(fp)
        end_time = time.time()
        print("Running time : ", end_time - start_time)

