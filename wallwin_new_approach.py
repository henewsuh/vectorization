import time
import numpy as np
from PIL import Image
import queue
import os
import scipy as sc
import geojson as gj
import cv2
import networkx as nx
import geopandas as gpd
import fiona
from matplotlib import pyplot as plt
from shapely.geometry import shape
import geojson
from shapely.geometry import Point as point
from shapely.geometry import LineString as string
from shapely.affinity import affine_transform as af
from shapely.geometry import Polygon
from math import dist
from geojson import Feature, FeatureCollection, dump
import shapely as sp
from shapely.ops import polygonize, unary_union, snap
from tqdm import tqdm
from tqdm import trange 
import sys
from osgeo import gdal
from osgeo import osr
from skimage.morphology import thin
from skimage import img_as_ubyte
from shapely.geometry import MultiLineString
from shapely.ops import linemerge
import scipy.spatial as spatial
from shapely.geometry import Point
from shapely.geometry import LineString


'''
1. raster import (o)
2. sk (o)
3. wall & window overlay and classify sk (o)
4-1. make wall sk (raster) as vec (o)
4-2. find wall end points / get their coords (o)
5. contour over the window raster (o)
6. grouping cd coords by window contour polygons
7. calculate the distance between wall and win coords 
8. change win coords to their adj wall coords 
9. make sett 
10. generate links (edges) between win coords (converted one)
'''

def thinning(filepath, fpnumber, fpnumberr, outputpath, separated_path, srcc):
    path_dir = separated_path
    os.chdir(path_dir)
    
    img_name = fpnumber + srcc
    img = cv2.imread(img_name)
    
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
    
    if srcc.endswith('.png'):
        src = srcc[:-4]
        
    cv2.imwrite(outputpath + '/' + fpnumber + src + '_thin.bmp' , BW_Skeleton)
    return BW_Skeleton

def overlay(filepath, fpnumber, fpnumberr, outputpath, separated_path): 
        
    overlay = cv2.imread(outputpath + '/' + fpnumber + '_merged_thin.bmp')        
    wall_background = cv2.imread(separated_path + fpnumber + "_wall.png")
    window_background = cv2.imread(separated_path + fpnumber + "_window.png")
    
    aa = overlay + wall_background
    bb = overlay + window_background 
    
    
    os.chdir(outputpath)
    canvas_wall = np.zeros(shape=overlay.shape, dtype=np.uint8)
    canvas_wall.fill(255)
    canvas_wall[np.where((aa == [131,232,255]).all(axis = 2))] = [0,0,0] #wall
    cv2.imwrite(fpnumber + '_wall_thin.bmp', canvas_wall)
    
    
    canvas_window = np.zeros(shape=overlay.shape, dtype=np.uint8)
    canvas_window.fill(255)
    canvas_window[np.where((bb == [0,0,255]).all(axis = 2))] = [0,0,0] #
    cv2.imwrite(fpnumber + '_window_thin.bmp', canvas_window)

def binarization(filepath, fpnumber, fpnumberr, outputpath, src): 
    
    img = cv2.imread(outputpath + fpnumber + src + '_thin.bmp')
    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)  # 이진화 시행
    cons = ~binary
    
    cv2.imwrite(outputpath + fpnumber + src + '_bin.bmp', cons)

def line_generation(mainpath, outputpath, fpnumber, fpnumberr, src):
    
    def r2v(outputpath, inimage, fpnumber, src):  
        if fpnumber.endswith('.png'):
            fpnumberr = fpnumber[:-4]  
        else: 
            fpnumberr = fpnumber       
        outresult = inimage.split(".")[0]+".geojson"
        
        if 'win' in str(inimage):
            os.chdir(outputpath)
            outresult = str(fpnumberr) + '_window.geojson'
        elif 'wall' in str(inimage):
            os.chdir(outputpath)
            outresult = str(fpnumberr) + '_wall.geojson'
        elif 'stair' in src:
            os.chdir(outputpath)
            outresult = str(fpnumberr) + '_stair.geojson'
        else:
            os.chdir(outputpath)
            outresult = str(fpnumberr) + '_lift.geojson'
        
        src_ds = gdal.Open(outputpath + fpnumber + src) #read image
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
            points.append((cX, cY))
       
        maxD = np.sqrt(2.0*pixelWidth*pixelWidth)
        tree = spatial.cKDTree(points)
        groups = tree.query_ball_tree(tree, maxD)
        
        out_lines = []
        for x, ptlist in enumerate(groups):
            for pt in ptlist:
                if pt>x:            
                    out_lines.append(LineString([Point(points[x]), Point(points[pt])]))    
        
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
        if isinstance(merged_lines,LineString):
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
            os.chdir(outputpath)
            geojson.dump(feature_collection, outfile)
        return feature_collection

    
    if src == None :
        #wall line generation
        wall_file_name = str(fpnumber) + '_wall_bin.bmp'
        src_wall = '_wall_bin.bmp'
        wall_vec = r2v(outputpath, wall_file_name, fpnumber, src_wall)
        wall_vec = [shape(i['geometry']) for i in wall_vec['features']]
        
        #window line generation
        win_file_name = str(fpnumber) + '_window_bin.bmp'
        src_window = '_window_bin.bmp'
        win_vec = r2v(outputpath, win_file_name, fpnumber, src_window)
        if win_vec:
            win_vec = [shape(i['geometry']) for i in win_vec['features']]
            with open(outputpath + fpnumber + "_wall.geojson") as f:
                linevec = geojson.load(f)
            linevec = linevec['features']
            linevec = [shape(linevec[i]['geometry']) for i in range(len(linevec))]  #convert geojson obj to shapely obj   
        else:
           win_vec = False 
           return
        os.chdir(outputpath)
    
    if src != None: 
        #stair & lift   
        file_name = str(fpnumberr) + '_' + src + '_bin.bmp'
        srccc = '_' + src + '_bin.bmp'
        
        line_vec = r2v(outputpath, file_name, fpnumberr, srccc)
        line_vec = [shape(i['geometry']) for i in line_vec['features']]

def sp_corner(outputpath, fpnumber, fpnumberr, src):

    img = cv2.imread(outputpath + fpnumber + src + '_bin.bmp', 0)
    height, width = img.shape
 
    #_wall or _window = src 
    with open(outputpath + fpnumber + src + '.geojson') as f:
        vec = gj.load(f)
    vec_line = [vec['features'][i]['geometry'] for i in range(len(vec['features']))]#geojson obj
    vec_line = [shape(vec_line[i]) for i in range(len(vec_line))] #shapely obj
    
    affine_m = [1, 0, 0, -1, 0, width]
    sp_coord = []
    
    for i in range(len(vec_line)):
        final_line = af(vec_line[i], affine_m)
        if len(final_line.boundary) != 0:
            start, ends = final_line.boundary #각 라인의 양끝점 좌표를 shapely.geometry.point.Point 형태로 받음 
            start_x = int(start.xy[0][0])
            start_y = int(start.xy[1][0])
            ends_x = int(ends.xy[0][0])
            ends_y = int(ends.xy[1][0])
            
            sp_coord.append([start_x, start_y])
            sp_coord.append([ends_x, ends_y])
    
    return sp_coord

def contour_polygon(separated_path, fpnumber, fpnumberr, outputpath): 
    
    filename = separated_path + fpnumber
    img = cv2.imread(filename + '_window.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    test = binary

    
    contours, h = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    #cv2.imwrite('bin_' + str(dong) + '.png', image)

    canvas = np.zeros(shape=img.shape, dtype=np.uint8)
    canvas.fill(255)
    canvas[np.where((image == [0,0,0]).all(axis = 2))] = [255,255,255]
    canvas[np.where((image == [0,0,255]).all(axis = 2))] = [0,0,0]
    canvas[np.where((image == [0,255,0]).all(axis = 2))] = [0,255,0]
    cv2.imwrite(outputpath + fpnumber + '_window_contour.png', canvas)
    
    poly_objs = []
    for i in range(len(contours)):
        if (i > 0) and (len(contours[i])) > 2:
            poly_objs.append(Polygon(np.squeeze(contours[i])).buffer(1))
    
    return poly_objs

def nodes_and_short_distance(wall_coords, window_coords):
    
    note = []
    sett = [] 
    
    for i in range(len(window_coords)):
        d = dict() 
        for j in range(len(wall_coords)):
            window_x = window_coords[i][0]
            window_y = window_coords[i][1]
            wall_x = wall_coords[j][0]
            wall_y = wall_coords[j][1] 
            d[dist((window_x, window_y), (wall_x, wall_y))] = (wall_x, wall_y)
        min_d = 99999
        for a in d.keys(): #shortest distance keeper 
            if a < min_d:
                min_d = a    
        
        if min_d < 15: 
            sett.append([[window_x, window_y], [d[min_d][0], d[min_d][1]], [min_d]])
    
    return sett

def corner_translation(filepath, outputpath, wall_n_window_pairs, fpnumber, fpnumberr, separated_path):
    
    img = cv2.imread(separated_path + fpnumber + '_merged.png')
    
    new_pair = [] #인접 벽의 좌표로 창문의 코너 좌표를 변환
    for i in range(len(wall_n_window_pairs)):
        new_window_x = wall_n_window_pairs[i][1][0] #change window x-coord to nearest wall x-coord         
        new_window_y = wall_n_window_pairs[i][1][1] #change window y-coord to nearest wall y-coord  
        ori_window_x = wall_n_window_pairs[i][0][0] #keep the original window x-coord
        ori_window_y = wall_n_window_pairs[i][0][1] #keep the original window y-coord
        new_pair.append([[ori_window_x, ori_window_y], [new_window_x, new_window_y], 
                         [wall_n_window_pairs[i][1][0], wall_n_window_pairs[i][1][1]], [wall_n_window_pairs[i][2][0]]])
        cv2.circle(img, (wall_n_window_pairs[i][1][0] , wall_n_window_pairs[i][1][1]),3 , 0 , 2)
        
    # cv2.imwrite(outputpath + fpnumber + '_window_ori2adj.png', img)
    return new_pair

def connect(separated_path, outputpath, fpnumber, fpnumberr, window_coords, wall_coords, window_polygon, new_pair):
        
    filename = separated_path + fpnumber
    img = cv2.imread(filename + '_merged.png')
    
    affine_m = [1, 0, 0, -1, 0, img.shape[0]]
    window_point = [] #change window_coords to shapley Points 
    for i in range(len(window_coords)): 
        window_x = window_coords[i][0]
        window_y = window_coords[i][1]
        a = point(window_x, window_y)
        apoint = af(a, affine_m)
        window_point.append(apoint) # calibrate coordinates using affine matrix 
        # window_point = shapely points of window endpoints 
    
    wall_point = [] 
    for i in range(len(wall_coords)): 
        wall_x = wall_coords[i][0]
        wall_y = wall_coords[i][1]
        apoint = af(point(wall_x, wall_y), affine_m)
        wall_point.append(apoint)
    
    wdict = dict()
    for i in range(len(window_polygon)):
        wlist = [] 
        for j in range(len(window_point)): 
            if window_polygon[i].contains(window_point[j]) == True: 
                wlist.append(j)
        wdict[i] = wlist
        # ddict = window_corners in the same window polygon
    
    def vis(window_polygon, window_point, wall_point):
        window_polygon = gpd.GeoDataFrame(window_polygon, columns = ['geometry'])
        window_point = gpd.GeoDataFrame(window_point, columns = ['geometry'])
        wall_point = gpd.GeoDataFrame(wall_point, columns = ['geometry'])
        window_polygon.to_file(outputpath + fpnumber + '_window_polys.shp')
        window_point.to_file(outputpath + fpnumber + '_window_corner.shp')
        wall_point.to_file(outputpath + fpnumber + '_wall_corner.shp')
    
    vis(window_polygon, window_point, wall_point)
    
    
    close = dict() # 폐합하는 부분 
    for i in range(len(wdict)):
        av = [] 
        for j in range(len((wdict[i]))): 
            idx = wdict[i][j] #window 코너의 인덱스
            crd = window_coords[idx] #해당 인덱스의 좌표값    
            for k in range(len(new_pair)): #new_set = [문원래좌표, 바뀐거, 인접벽, 거리]
                if crd == new_pair[k][0]: #문원래 좌표랑 맞는게 있으면 
                    crd_n = new_pair[k][1] #바뀐 좌표로 바꿔라
                    av.append(crd_n)
        close[i] = av
    
    # 인접 벽의 좌표들로 바뀐 문 좌표들을 문 인덱스별로 묶음
    window_line = [] 
    for i in range(len(close)):
        sng = [] 
        for j in range(len(close[i])):
            ax = close[i][j][0]
            ay = close[i][j][1]
            a = point(ax, ay)
            apoint = af(a, affine_m) 
            sng.append(apoint)
        if len(sng) > 1 and len(sng) < 9: 
            line = string(sng)
            window_line.append(line)
            
    
    window_gj = [] # window_line을 geojson으로 저장
    for i in range(len(window_line)):
        window_gj.append(Feature(geometry=window_line[i], properties={}))
    window = FeatureCollection(window_gj)
    with open(outputpath + fpnumber + '_window_line.geojson', 'w') as f:
        dump(window, f)
    
    with open(outputpath + fpnumber + '_wall.geojson') as f:
        wall_v = gj.load(f)
    wall_vecs = [wall_v['features'][i]['geometry'] for i in range(len(wall_v['features']))]#geojson obj
    wall_vecs = [shape(wall_vecs[i]) for i in range(len(wall_vecs))]#shapely obj
    wall_lines = sp.ops.linemerge(wall_vecs)
    # wall_poly = sp.ops.polygonize(wall_lines)
    
    
    ksk = []
    for i in range(len(window_line)):
        kiki = [] 
        for j in range(len(wall_lines)): #여기 수정이 필요할듯
            new_window_line = snap(window_line[i], wall_lines[j], tolerance = 2) #벽 라인에 대해서 문 라인을 스냅핑
            if window_line[i] == new_window_line: 
                continue
            else: 
                kiki.append(new_window_line)
        ksk.append(kiki)
                
    new_window_gj = [] #폐합하는 window
    for i in range(len(ksk)):
        for j in range(len(ksk[i])):
            new_window_gj.append(Feature(geometry=ksk[i][j], properties={}))
        
    new_window = FeatureCollection(new_window_gj)
    with open(outputpath + fpnumber + '_window_new_line.geojson', 'w') as f:
        dump(new_window, f)
    
    return window_gj

# ===========================================================================================================                  
mainpath = 'C:/Users/user/Desktop/module3_vec/module3/'   
filepath = 'C:/Users/user/Desktop/module3_vec/module3/test_input/'
outputpath = 'C:/Users/user/Desktop/module3_vec/module3/test_output/new_approach/'
separated_path = 'C:/Users/user/Desktop/module3_vec/module3/separated_output/'
fpnumber = '19-2'
fpnumberr = fpnumber + '.png'   
# =========================================================================================================== 

img = cv2.imread(filepath + fpnumber)
wall_img = cv2.imread(separated_path + fpnumber + '_wall.png')
print("wall")
win_img = cv2.imread(separated_path + fpnumber + '_window.png')
print("window")


print("thinning start")
thinning(filepath, fpnumber, fpnumberr, outputpath, separated_path, srcc='_merged.png')
# thinning(filepath, fpnumber, fpnumberr, outputpath, separated_path, srcc='_window.png')

print("overlay start")
overlay(filepath, fpnumber, fpnumberr, outputpath, separated_path)

print("binarization start")
binarization(filepath, fpnumber, fpnumberr, outputpath, src='_wall')
binarization(filepath, fpnumber, fpnumberr, outputpath, src='_window')

print("r2v")
line_generation(mainpath, outputpath, fpnumber, fpnumberr, src=None)

print("start & end points")
wall_coords = sp_corner(outputpath, fpnumber, fpnumberr, src='_wall')
window_coords = sp_corner(outputpath, fpnumber, fpnumberr, src='_window')

print("window polygonization")
window_polygon = contour_polygon(separated_path, fpnumber, fpnumberr, outputpath)

print("distance calculation")
wall_n_window_pairs = nodes_and_short_distance(wall_coords, window_coords)

print("calibarating window coords to adj. wall coords")
new_pair = corner_translation(filepath, outputpath, wall_n_window_pairs, fpnumber, fpnumberr, separated_path)

print("make new window")
window_gj = connect(separated_path, outputpath, fpnumber, fpnumberr, window_coords, wall_coords, window_polygon, new_pair) 














































