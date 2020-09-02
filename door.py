import shapefile
import os
import numpy as np
import cv2
from collections import defaultdict
import sys
from math import dist
import time
import pickle 
import networkx as nx
from shapely.geometry import Polygon, Point, mapping, LineString, shape
from shapely.affinity import affine_transform as af
import geopandas as gpd
from geojson import Feature, FeatureCollection, dump
import geojson as gj
import shapely as sp
from shapely.ops import polygonize, unary_union, snap



def wall_corner_extraction_harris(outputpath, fpnumber, separated_path):
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]
    else: 
        fpnumberr = fpnumber 
        
    filename = outputpath + fpnumberr + '/' +  fpnumberr + '_thin.bmp'
    img = cv2.imread(filename, 0)
    height, width = img.shape
    original = img.copy()
    
    dst = cv2.cornerHarris(img,2,3,0.04)

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
    corner_harris = []
    
    for i,p in enumerate(res):
        x,y = p[2:4]
        corner_harris.append([x,height-y])
        cv2.circle(img,(x,y),3 , 0 , 2)
    cv2.imwrite(outputpath + fpnumberr + '/' + fpnumberr + '_wall_corner.png' , img)
    return corner_harris # [id of node , x, y]

def pre_door(separated_path, fpnumber, outputpath):
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]
    else: 
        fpnumberr = fpnumber 
        
    img = cv2.imread(separated_path + fpnumberr + '_door.png', 0) #blue laye
    img[img != 255] = 0
    cv2.imwrite(outputpath + '/' + fpnumber+ '/'  + fpnumberr + '_door.bmp' , img)

def countblack(px,py,img,size):
    h1 = int(size/2)
    h2 = size - h1
    rect = img[py-h1:py+h2,px-h1:px+h2]
    dot = rect[rect == 0 ]
    return float(len(dot))/(size*size)


def door_corner_extraction(outputpath , fpnumber):
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]
    else: 
        fpnumberr = fpnumber 
        
    #import image
    filename = outputpath + fpnumberr + '/' + fpnumberr + '_door.bmp' 
    img = cv2.imread(filename,0)
    #image size
    height, width = img.shape
    original = img.copy()

    
    #Harris corner detection
    dst = cv2.cornerHarris(img,2,7,0.05)

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
        
    cv2.imwrite(outputpath + fpnumberr + '/'  + fpnumberr + '_door_corner.png' , img)
    corner_harris = np.asarray(corner_harris)

    return corner_harris

def wall_nodes_extraction(nodes, window_coord_id):
    
    wall_coord = nodes.copy()
    all_nodes = nodes.copy()
    window_coord_id = window_coord_id
    
    for i in range(len(window_coord_id)):
        for j in range(len(all_nodes)):
            if window_coord_id[i] in wall_coord:
                del wall_coord[window_coord_id[i]]
    
#    wall_coord = wall_coord.values() #extract values only (x,y) coordinates
#    wall_coord = list(wall_coord) #make them list
    
    wall_coord_real = wall_coord.copy()
    
    w_nodes = dict()
    idx = 0
        
    wall_coord_ = dict()
    for k, v in wall_coord.items():
        wall_coord_[v] = k
        
    w_nodes = dict()
    for k, v in wall_coord_.items():
        w_nodes[v] = k
    
    
    return wall_coord_real, wall_coord_


def harris_tolist(harris_corner):
    corner = harris_corner
    clist = corner.tolist()
    ccoord = [] 
    for i in range(len(clist)):
        x = clist[i][1]
        y = clist[i][2]
        ccoord.append([x,y]) 
    
    return ccoord

def nodes_and_short_distance(wall_harris_coord, door_harris_coord):
    note = []
    sett = [] 
    for i in range(len(door_harris_coord)):
        d = dict() 
        for j in range(len(wall_harris_coord)):
            door_x = door_harris_coord[i][0]
            door_y = door_harris_coord[i][1]
            wall_x = wall_harris_coord[j][0]
            wall_y = wall_harris_coord[j][1] 
            d[dist((door_x, door_y), (wall_x, wall_y))] = (wall_x, wall_y)
        min_d = 99999
        for a in d.keys(): #shortest distance keeper 
            if a < min_d:
                min_d = a           
        # if min_d < 30:  #######여기 threshold 조절 필요!!!!!!!!!
        #     sett.append([[door_x, door_y], [d[min_d][0], d[min_d][1]], [min_d]])        
        sett.append([[door_x, door_y], [d[min_d][0], d[min_d][1]], [min_d]])
        
    
    return sett

def contour(separated_path, fpnumber): 
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]
    else: 
        fpnumberr = fpnumber 
        
    filename = separated_path + fpnumber 
    img = cv2.imread(filename + '_door.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    test = binary

    
    contours, h = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    image = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    #cv2.imwrite('bin_' + str(dong) + '.png', image)

    canvas = np.zeros(shape=img.shape, dtype=np.uint8)
    canvas.fill(255)
    canvas[np.where((image == [0,0,0]).all(axis = 2))] = [255,255,255]
    canvas[np.where((image == [0,255,0]).all(axis = 2))] = [0,255,0]
    cv2.imwrite(outputpath + fpnumberr + '/' + fpnumberr + '_door_contour.png', canvas)
    
    poly_objs = []
    for i in range(len(contours)):
        if len(contours[i]) >= 1:
            poly_objs.append(Polygon(np.squeeze(contours[i])).buffer(3))
    
    return poly_objs

def corner_translation(filepath, outputpath, sett, fpnumber):
    if fpnumber.endswith('.png'):
        fpnumberr = fpnumber[:-4]
    else: 
        fpnumberr = fpnumber
        
    new_set = [] #인접 벽의 좌표로 문의 코너 좌표를 변환
    for i in range(len(sett)):
        new_door_x = sett[i][1][0] #change door x-coord to nearest wall x-coord         
        new_door_y = sett[i][1][1] #change door y-coord to nearest wall y-coord  
        ori_door_x = sett[i][0][0] #keep the original door x-coord
        ori_door_y = sett[i][0][1] #keep the original door y-coord
        new_set.append([[ori_door_x, ori_door_y], [new_door_x, new_door_y], [sett[i][1][0], sett[i][1][1]], [sett[i][2][0]]])
        cv2.circle(img,(sett[i][1][0] , sett[i][1][1]),3 , 0 , 2)
    cv2.imwrite(outputpath + fpnumberr + '/' + fpnumberr + '_door_ori2adj.png', img)
    
    return new_set

def vis(door_polys, door_points, wall_points):
    door_polys = gpd.GeoDataFrame(door_polys, columns = ['geometry'])
    door_points = gpd.GeoDataFrame(door_points, columns = ['geometry'])
    wall_points = gpd.GeoDataFrame(wall_points, columns = ['geometry'])
    door_polys.to_file('C:/Users/user/Desktop/door_polys.shp')
    door_points.to_file('C:/Users/user/Desktop/door_points.shp')
    wall_points.to_file('C:/Users/user/Desktop/raster_wall_points.shp')

def sp_corner(wall_line, img_width):
    
    affine_m = [1, 0, 0, -1, 0, img_width]
    shapely_wall_coord = []
    
    for i in range(len(wall_line)):
        wall_linee = af(wall_line[i], affine_m)
        start, ends = wall_linee.boundary #각 라인의 양끝점 좌표를 shapely.geometry.point.Point 형태로 받음 
        start_x = int(start.xy[0][0])
        start_y = int(start.xy[1][0])
        ends_x = int(ends.xy[0][0])
        ends_y = int(ends.xy[1][0])
        
        shapely_wall_coord.append([start_x, start_y])
        shapely_wall_coord.append([ends_x, ends_y])
    
    return shapely_wall_coord
        
# ========================================================================================================


filepath = 'C:/Users/user/Desktop/module3_vec/module3/test_input/'
outputpath = 'C:/Users/user/Desktop/module3_vec/module3/test_output/'
separated_path = 'C:/Users/user/Desktop/module3_vec/module3/separated_output/'
fpnumber = '1-1'
img = cv2.imread('C:/Users/user/Desktop/module3_vec/module3/separated_output/1-1_merged.png', 0)
img_width = img.shape[0]
img_height = img.shape[1]


door_harris = door_corner_extraction(outputpath, fpnumber)
wall_harris_coord = wall_corner_extraction_harris(outputpath, fpnumber, separated_path)
door_harris_coord = harris_tolist(door_harris)



#####

# import wall.geojson as shapely objects
with open(outputpath + fpnumber + '/' + fpnumber + '_wall.geojson') as f:
    wall_v = gj.load(f)
wall_line = [wall_v['features'][i]['geometry'] for i in range(len(wall_v['features']))]#geojson obj
wall_line = [shape(wall_line[i]) for i in range(len(wall_line))] #shapely obj

shapely_wall_coord = sp_corner(wall_line, img_width)

#####

sett = nodes_and_short_distance(shapely_wall_coord, door_harris_coord)
new_set = corner_translation(filepath, outputpath, sett, fpnumber)

door_polygon = contour(separated_path, fpnumber) #make each door as polygon using contour 



affine_m = [1, 0, 0, -1, 0, img.shape[0]]
door_point = [] #change door_harris_coordinates to shapley Points 
for i in range(len(door_harris_coord)): 
    door_x = door_harris_coord[i][0]
    door_y = door_harris_coord[i][1]
    a = Point(door_x, door_y)
    point = af(a, affine_m)
    door_point.append(point) # calibrate coordinates using affine matrix 
    # door_point = shapely points of door corners 

wall_point = [] 
for i in range(len(shapely_wall_coord)): 
    wall_x = shapely_wall_coord[i][0]
    wall_y = shapely_wall_coord[i][1]
    point = af(Point(wall_x, wall_y), affine_m)
    wall_point.append(point)

ddict = dict()
for i in range(len(door_polygon)):
    dlist = [] 
    for j in range(len(door_point)): 
        if door_polygon[i].contains(door_point[j]) == True: 
            dlist.append(j)
    ddict[i] = dlist
    # ddict = door_corners in the same door polygon

vis(door_polygon, door_point, wall_point)


close = dict() # 폐합하는 부분 
for i in range(len(ddict)):
    av = [] 
    for j in range(len((ddict[i]))): 
        idx = ddict[i][j] #door 코너의 인덱스
        crd = door_harris_coord[idx] #해당 인덱스의 좌표값    
        for k in range(len(new_set)): #new_set = [문원래좌표, 바뀐거, 인접벽, 거리]
            if crd == new_set[k][0]: #문원래 좌표랑 맞는게 있으면 
                crd_n = new_set[k][1] #바뀐 좌표로 바꿔라
                av.append(crd_n)
    close[i] = av

# 인접 벽의 좌표들로 바뀐 문 좌표들을 문 인덱스별로 묶음

door_line = [] 
for i in range(len(close)):
    sng = [] 
    for j in range(len(close[i])):
        ax = close[i][j][0]
        ay = close[i][j][1]
        a = Point(ax, ay)
        point = af(a, affine_m) 
        sng.append(point)
    if len(sng) > 1 and len(sng) < 9: 
        line = LineString(sng)
        door_line.append(line)
        
door_gj = [] # door_line을 geojson으로 저장
for i in range(len(door_line)):
    door_gj.append(Feature(geometry=door_line[i], properties={}))
door = FeatureCollection(door_gj)
with open(outputpath + fpnumber + '/' + fpnumber + '_door_line.geojson', 'w') as f:
    dump(door, f)



with open(outputpath + fpnumber + '/' + fpnumber + '_wall.geojson') as f:
    wall_v = gj.load(f)
wall_vecs = [wall_v['features'][i]['geometry'] for i in range(len(wall_v['features']))]#geojson obj
wall_vecs = [shape(wall_vecs[i]) for i in range(len(wall_vecs))]#shapely obj
wall_lines = sp.ops.linemerge(wall_vecs)
# wall_poly = sp.ops.polygonize(wall_lines)

ksk = []
for i in range(len(door_line)):
    kiki = [] 
    for j in range(len(wall_lines)): #여기 수정이 필요할듯
        new_door_line = snap(door_line[i], wall_lines[j], tolerance = 2) #벽 라인에 대해서 문 라인을 스냅핑
        if door_line[i] == new_door_line: 
            continue
        else: 
            kiki.append(new_door_line)
    ksk.append(kiki)
            
new_door_gj = [] #폐합하는 door
for i in range(len(ksk)):
    for j in range(len(ksk[i])):
        new_door_gj.append(Feature(geometry=ksk[i][j], properties={}))
    
new_door = FeatureCollection(new_door_gj)
with open(outputpath + fpnumber + '/' + fpnumber + '_door_new_line.geojson', 'w') as f:
    dump(new_door, f)

# # import wall.geojson as shapely objects
# with open(outputpath + fpnumber + '/' + fpnumber + '_wall.geojson') as f:
#     wall_v = gj.load(f)
# wall_line = [wall_v['features'][i]['geometry'] for i in range(len(wall_v['features']))]#geojson obj
# wall_line = [shape(wall_vecs[i]) for i in range(len(wall_vecs))] #shapely obj

# asdfasd = sp_corner(wall_line)